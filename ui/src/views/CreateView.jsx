/** Speech composition workspace for HTTP, saved, and live generation. */

import {
  AudioWaveform,
  ChevronDown,
  CirclePlus,
  LoaderCircle,
  Radio,
  Save,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api/client";
import { SpeechStreamClient } from "../api/stream";
import { AudioPlayer } from "../components/AudioPlayer";
import { ProfileFormModal } from "../components/ProfileFormModal";
import { useApp } from "../state/AppContext";
import { useToast } from "../state/ToastContext";
import { problemMessage } from "../utils/format";

const LANGUAGES = [
  ["en", "English"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
  ["de", "German"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["ru", "Russian"],
  ["auto", "Automatic"],
];

/** Format canonical model speaker IDs for display. */
function speakerName(value) {
  return value.split("_").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
}

/** Render voice, text, instruction, model, and generation controls. */
export function CreateView({ navigate }) {
  const {
    models,
    profiles,
    refreshProfiles,
    selectedProfileId,
    setSelectedProfileId,
    setTtsModel,
    startModelTransition,
    ttsModel,
  } = useApp();
  const { showToast } = useToast();
  const streamRef = useRef(new SpeechStreamClient());
  const localUrlRef = useRef(null);
  const [text, setText] = useState("");
  const [instructions, setInstructions] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [language, setLanguage] = useState("en");
  const [seed, setSeed] = useState("");
  const [save, setSave] = useState(false);
  const [live, setLive] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [streamBytes, setStreamBytes] = useState(0);
  const [newVoiceOpen, setNewVoiceOpen] = useState(false);
  const [result, setResult] = useState(null);
  const selectedProfile = profiles.find((item) => item.id === selectedProfileId);
  const ttsModels = models.filter((model) => model.kind === "tts");
  const selectedModel = models.find((model) => model.id === ttsModel);
  const loadedModelId = ttsModels.find((model) => (
    model.backend_supported && model.load_state === "loaded"
  ))?.id;
  const builtin = selectedModel?.speech?.voice_mode === "builtin";
  const speakers = selectedModel?.speech?.speakers || [];
  const acceptsInstructions = selectedModel?.speech?.instructions === true;
  const validVoice = builtin
    ? speakers.includes(speaker)
    : selectedProfile?.sample_count > 0;
  const canGenerate = Boolean(
    validVoice
    && text.trim()
    && selectedModel?.load_state === "loaded"
    && !busy,
  );

  // Re-entering Create or completing a model load selects the ready model.
  // A manual dropdown change remains possible until the loaded model changes.
  useEffect(() => {
    if (loadedModelId) {
      setTtsModel(loadedModelId);
    }
  }, [loadedModelId, setTtsModel]);

  useEffect(() => () => {
    streamRef.current.close();
    if (localUrlRef.current) {
      URL.revokeObjectURL(localUrlRef.current);
    }
  }, []);

  const resultUrl = useMemo(() => result?.audioUrl || "", [result]);
  /** Replace the latest result while releasing the prior object URL. */
  const setBlobResult = (blob, generation = null) => {
    if (localUrlRef.current) {
      URL.revokeObjectURL(localUrlRef.current);
    }
    localUrlRef.current = URL.createObjectURL(blob);
    setResult({
      audioUrl: localUrlRef.current,
      blob,
      generation,
      text: generation?.input_text || text.trim(),
    });
  };

  /** Generate through REST or the persistent WebSocket transport. */
  const generate = async (event) => {
    event.preventDefault();
    if (!canGenerate) {
      return;
    }
    const payload = {
      ...(builtin ? { speaker } : { profile_id: selectedProfileId }),
      text: text.trim(),
      language,
      model: ttsModel,
      seed: seed === "" ? null : Number(seed),
      instructions: acceptsInstructions ? instructions.trim() || null : null,
    };
    setBusy(true);
    setStreamBytes(0);
    try {
      if (live) {
        const streamed = await streamRef.current.generate(
          { ...payload, store: save },
          { onChunk: (bytes) => setStreamBytes((total) => total + bytes) },
        );
        setBlobResult(streamed.blob, streamed.generation);
      } else if (save) {
        const generation = await api.createGeneration(payload);
        setResult({
          audioUrl: generation.audio_url,
          blob: null,
          generation,
          text: generation.input_text,
        });
      } else {
        setBlobResult(await api.generateSpeech(payload));
      }
      showToast(save ? "Speech generated and saved." : "Speech generated.", "success");
    } catch (error) {
      if (error.name !== "AbortError") {
        showToast(problemMessage(error), "error");
      }
    } finally {
      setBusy(false);
    }
  };

  /** Create a voice and move directly to its sample workspace. */
  const createProfile = async (name) => {
    try {
      const created = await api.createProfile(name);
      await refreshProfiles();
      setSelectedProfileId(created.id);
      showToast(`${created.name} was created. Add a sample next.`, "success");
      navigate("/voices");
    } catch (error) {
      showToast(problemMessage(error), "error");
      throw error;
    }
  };

  return (
    <div className="create-page">
      <section className="panel composer-panel">
        <div className="section-heading">
          <div>
            <h1>Create speech</h1>
          </div>
          {!builtin && <button
            className="button outline new-voice-button"
            onClick={() => setNewVoiceOpen(true)}
            type="button"
          >
            <CirclePlus size={17} />
            New voice
          </button>}
        </div>
        <form onSubmit={generate}>
          <label className="field">
            <span>{builtin ? "Built-in speaker" : "Voice"}</span>
            <select
              onChange={(event) => builtin
                ? setSpeaker(event.target.value)
                : setSelectedProfileId(event.target.value)}
              value={builtin ? speaker : selectedProfileId}
            >
              <option value="">{builtin ? "Choose a speaker" : "Choose a cloned voice"}</option>
              {builtin ? speakers.map((value) => (
                <option key={value} value={value}>{speakerName(value)}</option>
              )) : profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} · {profile.sample_count} samples
                </option>
              ))}
            </select>
          </label>
          {!builtin && !profiles.length && (
            <div className="empty-state">
              <h2>Start with a voice</h2>
              <p>Create a voice and add a reference sample, or choose CustomVoice below.</p>
            </div>
          )}
          <label className="field">
            <span>Text</span>
            <textarea
              maxLength={5000}
              onChange={(event) => setText(event.target.value)}
              placeholder="Type what you want this voice to say."
              rows={6}
              value={text}
            />
            <small>{text.length} / 5000</small>
          </label>
          {acceptsInstructions && <label className="field">
            <span>Instructions</span>
            <textarea
              maxLength={500}
              onChange={(event) => setInstructions(event.target.value)}
              placeholder="Warm, intimate, unhurried"
              rows={2}
              value={instructions}
            />
            <small>{instructions.length} / 500</small>
          </label>}
          <div className="field-grid">
            <label className="field">
              <span>Language</span>
              <select
                onChange={(event) => setLanguage(event.target.value)}
                value={language}
              >
                {LANGUAGES.map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Qwen model</span>
              <select
                onChange={(event) => {
                  setTtsModel(event.target.value);
                  setSpeaker("");
                  setInstructions("");
                }}
                value={ttsModel}
              >
                {ttsModels.map((model) => (
                  <option disabled={!model.backend_supported} key={model.id} value={model.id}>
                    {model.display_name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <button
            aria-expanded={advanced}
            className="advanced-toggle"
            onClick={() => setAdvanced((current) => !current)}
            type="button"
          >
            <SlidersHorizontal size={17} />
            Advanced
            <ChevronDown className={advanced ? "is-open" : ""} size={16} />
          </button>
          {advanced && (
            <div className="advanced-panel">
              <label className="field">
                <span>Seed <em>optional</em></span>
                <input
                  min="0"
                  onChange={(event) => setSeed(event.target.value)}
                  placeholder="Leave blank for random"
                  type="number"
                  value={seed}
                />
              </label>
              <label className="switch-row">
                <input
                  checked={live}
                  onChange={(event) => setLive(event.target.checked)}
                  type="checkbox"
                />
                <span className="switch" />
                <span>
                  <strong>Live WebSocket audio</strong>
                  <small>Play PCM chunks as the backend produces them.</small>
                </span>
              </label>
            </div>
          )}
          <div className="generate-row">
            <button
              className="button primary generate-button"
              disabled={!canGenerate}
              type="submit"
            >
              {busy
                ? <LoaderCircle className="spin" size={19} />
                : <AudioWaveform size={20} />}
              {busy ? "Generating…" : "Generate speech"}
            </button>
            {busy && live ? (
              <button
                className="button secondary"
                onClick={() => streamRef.current.cancel()}
                type="button"
              >
                Cancel
              </button>
            ) : (
              <label className="save-toggle">
                <input
                  checked={save}
                  onChange={(event) => setSave(event.target.checked)}
                  type="checkbox"
                />
                <span className="switch" />
                <Save size={16} />
                Save to library
              </label>
            )}
          </div>
          {busy && live && (
            <p className="stream-status">
              <Radio size={15} />
              Receiving live audio · {Math.round(streamBytes / 1024)} KB
            </p>
          )}
        </form>
      </section>

      <aside className="panel voice-context">
        <div className="voice-context-card">
          <Sparkles size={18} />
          <h2>{builtin
            ? speakerName(speaker) || "Choose a speaker"
            : selectedProfile?.name || "Choose a voice"}</h2>
          <p>{builtin
            ? "Built-in voice · no reference needed"
            : `${selectedProfile?.sample_count || 0} samples`}</p>
        </div>
        <div className="model-context">
          <span className={`status-line${selectedModel?.load_state === "loaded"
            ? " is-ready"
            : ""}`}
          >
            <i />
            {selectedModel?.load_state === "loaded" ? "Ready" : "Not loaded"}
          </span>
          <p>{selectedModel?.display_name || "Select a model"}</p>
          {selectedModel && selectedModel.load_state !== "loaded" && (
            <button
              className="button outline full"
              onClick={() => startModelTransition(selectedModel.id, "loaded")}
              type="button"
            >
              Load model
            </button>
          )}
        </div>
        {!builtin && <button
          className="text-button"
          onClick={() => navigate("/voices")}
          type="button"
        >
          Manage samples
        </button>}
      </aside>

      <section className="panel latest-result">
        <div className="section-heading compact">
          <div>
            <h2>{result?.text || "Your generated speech will appear here"}</h2>
          </div>
          {result?.generation && <span className="saved-badge">Saved</span>}
        </div>
        {resultUrl ? (
          <AudioPlayer
            downloadName={result?.generation
              ? `speech-${result.generation.id}.wav`
              : "generated-speech.wav"}
            src={resultUrl}
            title={result?.text}
          />
        ) : (
          <div className="result-placeholder">
            <AudioWaveform size={20} />
            <span>Generate speech to listen, scrub, and download it.</span>
          </div>
        )}
      </section>

      <ProfileFormModal
        onClose={() => setNewVoiceOpen(false)}
        onSubmit={createProfile}
        open={newVoiceOpen}
        submitLabel="Create voice"
        title="New voice"
      />
    </div>
  );
}
