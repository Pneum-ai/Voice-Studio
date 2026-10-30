/** Browser-only workspace for arranging and exporting custom voice scenes. */

import { Save } from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";

import { api } from "../api/client";
import { ClipInspector } from "../editor/components/ClipInspector";
import { SourceLibrary } from "../editor/components/SourceLibrary";
import { Timeline } from "../editor/components/Timeline";
import { Transport } from "../editor/components/Transport";
import {
  decodeEditorAudio,
  exportProject,
  previewProject,
  waveformPeaks,
} from "../editor/engine/audio_engine";
import {
  createClip,
  createEditorId,
  createHistory,
  EMPTY_PROJECT,
  historyReducer,
  projectDuration,
} from "../editor/state/project";
import {
  clearDraft,
  getAsset,
  loadDraft,
  putAsset,
  saveDraft,
} from "../editor/storage/drafts";
import { useApp } from "../state/AppContext";
import { useToast } from "../state/ToastContext";
import { problemMessage } from "../utils/format";
import { startWavRecording } from "../utils/recording";

/** Convert one stored generation into an editor source descriptor. */
function generationSource(generation) {
  return {
    id: `generation-${generation.id}`,
    kind: "generation",
    name: generation.input_text || "Saved generation",
    detail: generation.profile_name || "Library",
    duration: generation.duration_seconds,
    url: generation.audio_url,
    assetId: null,
  };
}

/** Convert one reference sample into an editor source descriptor. */
function sampleSource(sample, profile) {
  return {
    id: `sample-${sample.id}`,
    kind: "sample",
    name: sample.transcript || `${profile.name} reference sample`,
    detail: profile.name,
    duration: sample.duration_seconds,
    url: sample.audio_url,
    assetId: null,
  };
}

/** Render the complete local voice-scene editor. */
export function EditorView() {
  const { profiles } = useApp();
  const { showToast } = useToast();
  const [history, dispatch] = useReducer(historyReducer, undefined, createHistory);
  const [serverSources, setServerSources] = useState([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [hydrated, setHydrated] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [sourcesCollapsed, setSourcesCollapsed] = useState(true);
  const buffers = useRef(new Map());
  const stopPlayback = useRef(null);
  const stopRecording = useRef(null);
  const playbackClock = useRef(null);
  const project = history.present;
  const duration = projectDuration(project);

  const catalog = useMemo(() => {
    const known = new Set(serverSources.map((source) => source.id));
    return [
      ...project.sources.filter((source) => !known.has(source.id)),
      ...serverSources,
    ];
  }, [project.sources, serverSources]);
  const catalogById = useMemo(
    () => new Map(catalog.map((source) => [source.id, source])),
    [catalog],
  );
  const selectedClip = project.clips.find((clip) => clip.id === selectedId);
  const selectedSource = project.sources.find(
    (source) => source.id === selectedClip?.sourceId,
  );

  useEffect(() => {
    loadDraft()
      .then((draft) => {
        if (draft?.version === 1) {
          dispatch({ type: "hydrate", project: draft });
        }
      })
      .catch(() => showToast("Local draft recovery is unavailable.", "error"))
      .finally(() => setHydrated(true));
  }, [showToast]);

  useEffect(() => {
    if (!hydrated) {
      return undefined;
    }
    const timer = window.setTimeout(() => {
      saveDraft(project).catch(() => {
        showToast("The editor could not save this local draft.", "error");
      });
    }, 350);
    return () => window.clearTimeout(timer);
  }, [hydrated, project, showToast]);

  useEffect(() => {
    let active = true;
    setCatalogLoading(true);
    Promise.all([
      api.listGenerations({ limit: 100 }),
      ...profiles.map((profile) => api.listSamples(profile.id)),
    ])
      .then(([generations, ...sampleLists]) => {
        if (!active) {
          return;
        }
        const samples = sampleLists.flatMap((result, index) => (
          result.items.map((sample) => sampleSource(sample, profiles[index]))
        ));
        setServerSources([
          ...generations.items.map(generationSource),
          ...samples,
        ]);
      })
      .catch((error) => showToast(problemMessage(error), "error"))
      .finally(() => active && setCatalogLoading(false));
    return () => {
      active = false;
    };
  }, [profiles, showToast]);

  /** Resolve and cache one source's decoded audio. */
  const ensureDecoded = useCallback(async (source) => {
    if (buffers.current.has(source.id)) {
      return buffers.current.get(source.id);
    }
    const input = source.assetId ? await getAsset(source.assetId) : source.url;
    if (!input) {
      throw new Error(`${source.name} is no longer available in this browser.`);
    }
    const buffer = await decodeEditorAudio(input);
    buffers.current.set(source.id, buffer);
    return buffer;
  }, []);

  /** Decode every source currently used by timeline clips. */
  const ensureTimelineDecoded = useCallback(async () => {
    const sourceIds = new Set(project.clips.map((clip) => clip.sourceId));
    const sources = project.sources.filter((source) => sourceIds.has(source.id));
    await Promise.all(sources.map(ensureDecoded));
  }, [ensureDecoded, project.clips, project.sources]);

  /** Stop preview without changing the current edit decision list. */
  const pausePlayback = useCallback(async () => {
    const stop = stopPlayback.current;
    stopPlayback.current = null;
    if (stop) {
      await stop();
    }
    setPlaying(false);
  }, []);

  useEffect(() => () => {
    stopPlayback.current?.();
    stopRecording.current?.();
  }, []);

  useEffect(() => {
    if (!playing || !playbackClock.current) {
      return undefined;
    }
    let frame = 0;
    const update = () => {
      const { offset, startedAt } = playbackClock.current;
      setPlayhead(Math.min(duration, offset + (performance.now() - startedAt) / 1000));
      frame = window.requestAnimationFrame(update);
    };
    frame = window.requestAnimationFrame(update);
    return () => window.cancelAnimationFrame(frame);
  }, [duration, playing]);

  /** Add a catalog source to the timeline after decoding its waveform. */
  const addSource = async (sourceId, start = playhead) => {
    const source = catalogById.get(sourceId);
    if (!source) {
      return;
    }
    setBusy(true);
    try {
      const buffer = await ensureDecoded(source);
      const prepared = {
        ...source,
        duration: buffer.duration,
        peaks: source.peaks || waveformPeaks(buffer),
      };
      const clip = createClip(prepared, start);
      dispatch({ type: "execute", command: { type: "add_clip", source: prepared, clip } });
      setSelectedId(clip.id);
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setBusy(false);
    }
  };

  /** Store one local audio file and expose it in the source shelf. */
  const ingestLocalFile = async (file, kind = "upload") => {
    const buffer = await decodeEditorAudio(file);
    const id = createEditorId(kind);
    await putAsset(id, file);
    buffers.current.set(id, buffer);
    const source = {
      id,
      kind,
      name: file.name || "Microphone recording",
      detail: kind === "recording" ? "Recorded now" : "Uploaded audio",
      duration: buffer.duration,
      url: null,
      assetId: id,
      peaks: waveformPeaks(buffer),
    };
    dispatch({ type: "execute", command: { type: "add_source", source } });
  };

  /** Decode and retain all selected browser uploads. */
  const uploadFiles = async (files) => {
    if (!files.length) {
      return;
    }
    setBusy(true);
    try {
      for (const file of files) {
        await ingestLocalFile(file);
      }
      showToast(`${files.length} audio ${files.length === 1 ? "clip" : "clips"} added.`, "success");
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setBusy(false);
    }
  };

  /** Start or finish microphone capture and retain the resulting WAV. */
  const toggleRecording = async () => {
    try {
      if (!recording) {
        stopRecording.current = await startWavRecording();
        setRecording(true);
        return;
      }
      setBusy(true);
      const blob = await stopRecording.current();
      stopRecording.current = null;
      setRecording(false);
      const file = new File([blob], `recording-${Date.now()}.wav`, {
        type: "audio/wav",
      });
      await ingestLocalFile(file, "recording");
      showToast("Recording added to your clips.", "success");
    } catch (error) {
      stopRecording.current = null;
      setRecording(false);
      showToast(problemMessage(error), "error");
    } finally {
      setBusy(false);
    }
  };

  /** Toggle scheduled Web Audio preview from the current playhead. */
  const playPause = async () => {
    if (playing) {
      await pausePlayback();
      return;
    }
    setBusy(true);
    try {
      await ensureTimelineDecoded();
      const offset = playhead >= duration ? 0 : playhead;
      setPlayhead(offset);
      playbackClock.current = { offset, startedAt: performance.now() };
      const stop = previewProject(project, buffers.current, offset, async () => {
        stopPlayback.current = null;
        await stop();
        setPlaying(false);
        setPlayhead(duration);
      });
      stopPlayback.current = stop;
      setPlaying(true);
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setBusy(false);
    }
  };

  /** Render the current scene offline and download a WAV file. */
  const download = async () => {
    setBusy(true);
    await pausePlayback();
    try {
      await ensureTimelineDecoded();
      await exportProject(project, buffers.current);
      showToast("Voice scene exported as WAV.", "success");
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setBusy(false);
    }
  };

  /** Clear the edit decision list after explicit confirmation. */
  const resetProject = async () => {
    if (!window.confirm("Clear this voice scene? This cannot be undone.")) {
      return;
    }
    await pausePlayback();
    await clearDraft();
    dispatch({ type: "reset", project: EMPTY_PROJECT });
    setSelectedId(null);
    setPlayhead(0);
  };

  /** Execute one selected-clip patch. */
  const updateSelected = (changes) => {
    const next = { ...selectedClip, ...changes };
    const maxFade = Math.max(0, (next.outPoint - next.inPoint) / 2);
    dispatch({
      type: "execute",
      command: {
        type: "update_clip",
        clipId: selectedId,
        changes: {
          ...changes,
          fadeIn: Math.min(next.fadeIn, maxFade),
          fadeOut: Math.min(next.fadeOut, maxFade),
        },
      },
    });
  };

  return (
    <div className="editor-page">
      <section className="page-title-row editor-title-row">
        <div>
          <input
            aria-label="Project name"
            className="editor-project-name"
            onChange={(event) => dispatch({
              type: "execute",
              command: { type: "rename", name: event.target.value },
            })}
            value={project.name}
          />
          <p>Arrange saved, uploaded, and recorded audio into one custom scene.</p>
        </div>
        <span className="editor-draft-state"><Save size={14} /> Saved locally</span>
      </section>
      <Transport
        busy={busy}
        canRedo={Boolean(history.future.length)}
        canUndo={Boolean(history.past.length)}
        duration={duration}
        onClear={resetProject}
        onExport={download}
        onPlayPause={playPause}
        onRedo={() => dispatch({ type: "redo" })}
        onRestart={async () => {
          await pausePlayback();
          setPlayhead(0);
        }}
        onUndo={() => dispatch({ type: "undo" })}
        playing={playing}
        playhead={playhead}
      />
      <div className={[
        "editor-workspace",
        sourcesCollapsed ? "is-sources-collapsed" : "",
      ].filter(Boolean).join(" ")}>
        <SourceLibrary
          busy={busy}
          collapsed={sourcesCollapsed}
          items={catalog}
          loading={catalogLoading}
          onAdd={addSource}
          onRecord={toggleRecording}
          onToggle={() => setSourcesCollapsed((current) => !current)}
          onUpload={uploadFiles}
          recording={recording}
        />
        <div className="editor-main-column">
          <Timeline
            duration={duration}
            onDropSource={addSource}
            onMoveClip={(clipId, start) => dispatch({
              type: "execute",
              command: { type: "update_clip", clipId, changes: { start } },
            })}
            onSeek={async (time) => {
              await pausePlayback();
              setPlayhead(time);
            }}
            onSelect={setSelectedId}
            playhead={playhead}
            project={project}
            selectedId={selectedId}
          />
          <ClipInspector
            clip={selectedClip}
            onRemove={() => {
              dispatch({
                type: "execute",
                command: { type: "remove_clip", clipId: selectedId },
              });
              setSelectedId(null);
            }}
            onUpdate={updateSelected}
            source={selectedSource}
          />
        </div>
      </div>
    </div>
  );
}
