/** Shared API state and model lifecycle orchestration for every view. */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { api } from "../api/client";
import { followJob } from "../api/jobs";
import { problemMessage } from "../utils/format";
import { useToast } from "./ToastContext";

const AppContext = createContext(null);
/** Prefer loaded models, retaining a valid selection when none is loaded. */
export function preferredModel(models, kind, current) {
  const candidates = models.filter((model) => (
    model.kind === kind && model.backend_supported
  ));
  const loaded = candidates.find((model) => model.load_state === "loaded");
  if (loaded) {
    return loaded.id;
  }
  if (candidates.some((model) => model.id === current)) {
    return current;
  }
  return (
    loaded?.id
    || candidates[0]?.id
    || ""
  );
}

/** Load and provide profiles, models, jobs, health, and active selections. */
export function AppProvider({ children }) {
  const { showToast } = useToast();
  const [health, setHealth] = useState(null);
  const [profiles, setProfiles] = useState([]);
  const [models, setModels] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [ttsModel, setTtsModel] = useState("");
  const [sttModel, setSttModel] = useState("");
  const [loading, setLoading] = useState(true);
  const cleanups = useRef(new Set());

  /** Refresh service readiness without mutating model state. */
  const refreshHealth = useCallback(async () => {
    const result = await api.health();
    setHealth(result);
    return result;
  }, []);

  /** Refresh profiles and retain a valid voice selection. */
  const refreshProfiles = useCallback(async () => {
    const result = await api.listProfiles();
    setProfiles(result.items);
    setSelectedProfileId((current) => (
      result.items.some((profile) => profile.id === current)
        ? current
        : result.items[0]?.id || ""
    ));
    return result.items;
  }, []);

  /** Refresh models and retain compatible speech and transcription selections. */
  const refreshModels = useCallback(async () => {
    const result = await api.listModels();
    setModels(result.items);
    setTtsModel((current) => preferredModel(result.items, "tts", current));
    setSttModel((current) => (
      preferredModel(result.items, "transcription", current)
    ));
    return result.items;
  }, []);

  /** Refresh retained model lifecycle jobs. */
  const refreshJobs = useCallback(async () => {
    const result = await api.listJobs();
    setJobs(result.items);
    return result.items;
  }, []);

  /** Load all shared application resources concurrently. */
  const refreshAll = useCallback(async () => {
    setLoading(true);
    const results = await Promise.allSettled([
      refreshHealth(),
      refreshProfiles(),
      refreshModels(),
      refreshJobs(),
    ]);
    const failure = results.find((result) => result.status === "rejected");
    if (failure) {
      showToast(problemMessage(failure.reason), "error");
    }
    setLoading(false);
  }, [
    refreshHealth,
    refreshJobs,
    refreshModels,
    refreshProfiles,
    showToast,
  ]);

  useEffect(() => {
    const activeCleanups = cleanups.current;
    refreshAll();
    return () => {
      activeCleanups.forEach((cleanup) => cleanup());
      activeCleanups.clear();
    };
  }, [refreshAll]);

  /** Start a model transition and follow its event stream to completion. */
  const startModelTransition = useCallback(async (modelId, state) => {
    const job = await api.setModelState(modelId, state);
    setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
    return new Promise((resolve) => {
      let cleanup = null;
      cleanup = followJob(
        job.id,
        async (snapshot) => {
          setJobs((current) => [
            snapshot,
            ...current.filter((item) => item.id !== snapshot.id),
          ]);
          if (!["complete", "failed", "cancelled"].includes(snapshot.state)) {
            return;
          }
          cleanup?.();
          cleanups.current.delete(cleanup);
          await refreshModels();
          await refreshJobs();
          const complete = snapshot.state === "complete";
          const message = complete
            ? `${snapshot.resource_id} is ${state}.`
            : snapshot.error || `${snapshot.resource_id} ${snapshot.state}.`;
          showToast(message, complete ? "success" : "error");
          resolve(snapshot);
        },
        (error) => {
          cleanup?.();
          cleanups.current.delete(cleanup);
          showToast(problemMessage(error), "error");
          resolve({ state: "failed", error: problemMessage(error) });
        },
      );
      cleanups.current.add(cleanup);
    });
  }, [refreshJobs, refreshModels, showToast]);

  const value = useMemo(() => ({
    health,
    jobs,
    loading,
    models,
    profiles,
    refreshAll,
    refreshHealth,
    refreshJobs,
    refreshModels,
    refreshProfiles,
    selectedProfileId,
    setSelectedProfileId,
    setSttModel,
    setTtsModel,
    startModelTransition,
    sttModel,
    ttsModel,
  }), [
    health,
    jobs,
    loading,
    models,
    profiles,
    refreshAll,
    refreshHealth,
    refreshJobs,
    refreshModels,
    refreshProfiles,
    selectedProfileId,
    startModelTransition,
    sttModel,
    ttsModel,
  ]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

/** Access the required shared application state. */
export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useApp must be used inside AppProvider.");
  }
  return context;
}
