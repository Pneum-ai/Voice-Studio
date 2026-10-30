/** Voice profile and reference-sample management workspace. */

import {
  AudioLines,
  CirclePlus,
  Clock3,
  Mic2,
  MoreHorizontal,
  Pencil,
  Search,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import { AudioPlayer } from "../components/AudioPlayer";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { EmptyState } from "../components/EmptyState";
import { ProfileFormModal } from "../components/ProfileFormModal";
import { SampleFormModal } from "../components/SampleFormModal";
import { useApp } from "../state/AppContext";
import { useToast } from "../state/ToastContext";
import { formatDate, problemMessage } from "../utils/format";


/** Render searchable profiles and complete sample CRUD controls. */
export function VoicesView() {
  const {
    models,
    profiles,
    refreshProfiles,
    selectedProfileId,
    setSelectedProfileId,
    startModelTransition,
    sttModel,
  } = useApp();
  const { showToast } = useToast();
  const [query, setQuery] = useState("");
  const [detail, setDetail] = useState(null);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [profileModal, setProfileModal] = useState(null);
  const [sampleModal, setSampleModal] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const transcriptionModel = models.find((model) => model.id === sttModel);

  const filteredProfiles = useMemo(() => (
    profiles.filter((profile) => (
      profile.name.toLowerCase().includes(query.toLowerCase())
    ))
  ), [profiles, query]);

  /** Load one profile and its ordered reference samples. */
  const loadDetail = useCallback(async (profileId) => {
    if (!profileId) {
      setDetail(null);
      setSamples([]);
      return;
    }
    setLoading(true);
    try {
      const [profile, sampleList] = await Promise.all([
        api.getProfile(profileId),
        api.listSamples(profileId),
      ]);
      setDetail(profile);
      setSamples(sampleList.items);
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    loadDetail(selectedProfileId);
  }, [loadDetail, selectedProfileId]);

  /** Create and select a new name-only voice profile. */
  const createProfile = async (name) => {
    try {
      const created = await api.createProfile(name);
      await refreshProfiles();
      setSelectedProfileId(created.id);
      showToast(`${created.name} was created.`, "success");
    } catch (error) {
      showToast(problemMessage(error), "error");
      throw error;
    }
  };

  /** Rename the selected profile and refresh shared state. */
  const renameProfile = async (name) => {
    try {
      const updated = await api.renameProfile(selectedProfileId, name);
      await refreshProfiles();
      await loadDetail(updated.id);
      showToast("Voice renamed.", "success");
    } catch (error) {
      showToast(problemMessage(error), "error");
      throw error;
    }
  };

  /** Add a sample or update the selected sample transcript. */
  const saveSample = async ({ file, transcript, trim }) => {
    try {
      if (sampleModal?.mode === "edit") {
        await api.updateSample(
          selectedProfileId,
          sampleModal.sample.id,
          transcript,
        );
        showToast("Sample metadata updated.", "success");
      } else {
        await api.addSample(
          selectedProfileId,
          file,
          transcript,
          trim,
        );
        showToast("Reference sample added.", "success");
      }
      await refreshProfiles();
      await loadDetail(selectedProfileId);
    } catch (error) {
      showToast(problemMessage(error), "error");
      throw error;
    }
  };

  /** Automatically transcribe one pending reference sample. */
  const transcribeSample = async (file, trim, language = "") => (
    api.transcribe(file, sttModel, language, "", trim)
  );

  /** Explicitly load the selected model for sample transcription. */
  const loadTranscriptionModel = async () => {
    const result = await startModelTransition(sttModel, "loaded");
    if (result.state !== "complete") {
      throw new Error(result.error || "The transcription model could not be loaded.");
    }
  };

  /** Delete the currently confirmed profile or sample. */
  const confirmDelete = async () => {
    if (!deleteTarget) {
      return;
    }
    setDeleting(true);
    try {
      if (deleteTarget.kind === "profile") {
        await api.deleteProfile(deleteTarget.id);
        await refreshProfiles();
        showToast("Voice and its samples were deleted.", "success");
      } else {
        await api.deleteSample(selectedProfileId, deleteTarget.id);
        await refreshProfiles();
        await loadDetail(selectedProfileId);
        showToast("Reference sample deleted.", "success");
      }
      setDeleteTarget(null);
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="voices-page">
      <section className="panel voices-list-panel">
        <div className="section-heading compact">
          <div>
            <h1>Voices</h1>
          </div>
          <button
            aria-label="Create voice"
            className="icon-button accent"
            onClick={() => setProfileModal({ mode: "create" })}
            type="button"
          >
            <CirclePlus size={19} />
          </button>
        </div>
        <label className="search-field">
          <Search size={17} />
          <span className="sr-only">Search voices</span>
          <input
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search voices"
            value={query}
          />
        </label>
        <div className="voice-list" role="list">
          {filteredProfiles.map((profile) => (
            <button
              aria-current={profile.id === selectedProfileId}
              className={profile.id === selectedProfileId ? "is-active" : ""}
              key={profile.id}
              onClick={() => setSelectedProfileId(profile.id)}
              role="listitem"
              type="button"
            >
              <span className="voice-glyph"><AudioLines size={18} /></span>
              <span>
                <strong>{profile.name}</strong>
                <small>{profile.sample_count} samples</small>
              </span>
              <MoreHorizontal size={17} />
            </button>
          ))}
        </div>
      </section>

      <section className="panel voice-detail-panel">
        {!detail ? (
          <EmptyState
            action={(
              <button
                className="button primary"
                onClick={() => setProfileModal({ mode: "create" })}
                type="button"
              >
                <CirclePlus size={18} />
                New voice
              </button>
            )}
            description="Create a named profile and add at least one sample."
            icon={Mic2}
            title={loading ? "Loading voice…" : "No voice selected"}
          />
        ) : (
          <>
            <div className="voice-detail-header">
              <div>
                <h1>{detail.name}</h1>
                <p>
                  {detail.sample_count} reference samples · Updated{" "}
                  {formatDate(detail.updated_at)}
                </p>
              </div>
              <div className="action-row">
                <button
                  className="button outline"
                  onClick={() => setProfileModal({ mode: "rename" })}
                  type="button"
                >
                  <Pencil size={16} />
                  Rename
                </button>
                <button
                  aria-label={`Delete ${detail.name}`}
                  className="icon-button danger-text"
                  onClick={() => setDeleteTarget({
                    kind: "profile",
                    id: detail.id,
                    name: detail.name,
                  })}
                  type="button"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            </div>

            <div className="sample-heading">
              <div>
                <h2>Reference samples</h2>
                <p>Exact transcripts create more reliable cloned speech.</p>
              </div>
              <button
                className="button primary"
                onClick={() => setSampleModal({ mode: "add" })}
                type="button"
              >
                <CirclePlus size={17} />
                Add sample
              </button>
            </div>

            {samples.length ? (
              <div className="sample-list">
                {samples.map((sample, index) => (
                  <article className="sample-card" key={sample.id}>
                    <header>
                      <span className="sample-number">{index + 1}</span>
                      <div>
                        <strong>Sample {index + 1}</strong>
                        <small>
                          <Clock3 size={13} />
                          {sample.duration_seconds.toFixed(1)} sec ·{" "}
                          {sample.sample_rate / 1000} kHz
                        </small>
                      </div>
                      <div className="sample-actions">
                        <button
                          aria-label="Edit sample"
                          className="icon-button"
                          onClick={() => setSampleModal({
                            mode: "edit",
                            sample,
                          })}
                          type="button"
                        >
                          <Pencil size={16} />
                        </button>
                        <button
                          aria-label="Delete sample"
                          className="icon-button danger-text"
                          onClick={() => setDeleteTarget({
                            kind: "sample",
                            id: sample.id,
                            name: `Sample ${index + 1}`,
                          })}
                          type="button"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </header>
                    <p className="sample-transcript">{sample.transcript}</p>
                    <AudioPlayer
                      compact
                      downloadName={`${detail.name}-sample-${index + 1}.wav`}
                      src={sample.audio_url}
                      title={`${detail.name} sample ${index + 1}`}
                    />
                  </article>
                ))}
              </div>
            ) : (
              <EmptyState
                action={(
                  <button
                    className="button primary"
                    onClick={() => setSampleModal({ mode: "add" })}
                    type="button"
                  >
                    Add a reference sample
                  </button>
                )}
                description="Use a clean 2–30 second recording with its exact transcript."
                icon={AudioLines}
                title="This voice needs a sample"
              />
            )}
          </>
        )}
      </section>

      <ProfileFormModal
        initialName={profileModal?.mode === "rename" ? detail?.name : ""}
        onClose={() => setProfileModal(null)}
        onSubmit={profileModal?.mode === "rename" ? renameProfile : createProfile}
        open={Boolean(profileModal)}
        submitLabel={profileModal?.mode === "rename" ? "Save name" : "Create voice"}
        title={profileModal?.mode === "rename" ? "Rename voice" : "New voice"}
      />
      <SampleFormModal
        initialTranscript={sampleModal?.sample?.transcript || ""}
        mode={sampleModal?.mode || "add"}
        onClose={() => setSampleModal(null)}
        onLoadTranscriptionModel={loadTranscriptionModel}
        onSubmit={saveSample}
        onTranscribe={transcribeSample}
        open={Boolean(sampleModal)}
        transcriptionModel={transcriptionModel}
        transcriptionReady={transcriptionModel?.load_state === "loaded"}
      />
      <ConfirmDialog
        busy={deleting}
        description={deleteTarget?.kind === "profile"
          ? "Its reference samples and prompt cache will also be removed."
          : "The profile prompt cache will be rebuilt from the remaining samples."}
        onClose={() => setDeleteTarget(null)}
        onConfirm={confirmDelete}
        open={Boolean(deleteTarget)}
        title={`Delete ${deleteTarget?.name || "item"}?`}
      />
    </div>
  );
}
