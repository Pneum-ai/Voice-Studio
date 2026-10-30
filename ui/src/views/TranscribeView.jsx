/** Local transcription workspace for uploads and microphone recordings. */

import {
  AudioLines,
  Check,
  Clipboard,
  FileAudio,
  LoaderCircle,
  Upload,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import { AudioPlayer } from "../components/AudioPlayer";
import { MicrophoneRecorder } from "../components/MicrophoneRecorder";
import { useApp } from "../state/AppContext";
import { useToast } from "../state/ToastContext";
import { problemMessage } from "../utils/format";

/** Render upload preview, model options, and transcript results. */
export function TranscribeView() {
  const {
    models,
    setSttModel,
    startModelTransition,
    sttModel,
  } = useApp();
  const { showToast } = useToast();
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [language, setLanguage] = useState("");
  const [initialPrompt, setInitialPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [captureActive, setCaptureActive] = useState(false);
  const [copied, setCopied] = useState(false);
  const transcriptionModels = models.filter((model) => (
    model.kind === "transcription"
  ));
  const selectedModel = models.find((model) => model.id === sttModel);
  const capabilities = selectedModel?.transcription;

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (capabilities?.language_mode === "automatic") {
      setLanguage("");
    } else if (
      capabilities?.language_mode === "required"
      && !capabilities.languages.includes(language)
    ) {
      setLanguage(capabilities.languages[0] || "");
    }
  }, [capabilities, language]);

  const selectFile = useCallback((nextFile) => {
    setFile(nextFile);
    setResult(null);
    setCopied(false);
  }, []);

  /** Submit the selected file and supported recognition controls. */
  const transcribe = async (event) => {
    event.preventDefault();
    if (!file || busy || captureActive || !sttModel || selectedModel?.load_state !== "loaded") {
      return;
    }
    setBusy(true);
    setResult(null);
    try {
      const response = await api.transcribe(
        file,
        sttModel,
        capabilities?.language_mode === "automatic" ? null : language.trim() || null,
        capabilities?.context ? initialPrompt.trim() || null : null,
      );
      setResult(response);
      showToast("Transcription complete.", "success");
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setBusy(false);
    }
  };

  /** Copy the completed transcript to the browser clipboard. */
  const copyText = async () => {
    await navigator.clipboard.writeText(result.text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="transcribe-page">
      <section className="page-title-row">
        <div>
          <h1>Transcribe audio</h1>
          <p>Upload audio or record a microphone test, then choose a local recognition model.</p>
        </div>
        <span className={`status-line${selectedModel?.load_state === "loaded"
          ? " is-ready"
          : ""}`}
        >
          <i />
          {selectedModel?.load_state === "loaded" ? "Model ready" : "Model not loaded"}
        </span>
      </section>

      <div className="transcribe-grid">
        <form className="panel transcription-form" onSubmit={transcribe}>
          <label className={`file-drop large${file ? " has-file" : ""}`}>
            {file ? <FileAudio size={30} /> : <Upload size={30} />}
            <strong>{file ? file.name : "Drop or choose an audio file"}</strong>
            <span>
              {file
                ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                : "WAV, FLAC, MP3, M4A, and other common formats"}
            </span>
            <input
              accept="audio/*"
              aria-label="Choose an audio file"
              disabled={busy || captureActive}
              onChange={(event) => {
                if (event.target.files?.[0]) selectFile(event.target.files[0]);
                event.target.value = "";
              }}
              type="file"
            />
          </label>
          <MicrophoneRecorder
            disabled={busy}
            onActiveChange={setCaptureActive}
            onRecorded={selectFile}
          />
          {previewUrl && !captureActive && (
            <AudioPlayer
              compact
              downloadName={file.name}
              src={previewUrl}
              title={file.name}
            />
          )}
          <div className="field-grid">
            <label className="field">
              <span>Transcription model</span>
              <select
                onChange={(event) => setSttModel(event.target.value)}
                value={sttModel}
              >
                {transcriptionModels.map((model) => (
                  <option disabled={!model.backend_supported} key={model.id} value={model.id}>
                    {model.display_name}
                  </option>
                ))}
              </select>
            </label>
            {capabilities?.language_mode !== "automatic" && (
              <label className="field">
                <span>
                  Language {capabilities?.language_mode !== "required" && <em>optional</em>}
                </span>
                {capabilities?.languages?.length ? (
                  <select
                    onChange={(event) => setLanguage(event.target.value)}
                    required={capabilities.language_mode === "required"}
                    value={language}
                  >
                    {capabilities.language_mode !== "required" && (
                      <option value="">Automatic detection</option>
                    )}
                    {capabilities.languages.map((code) => (
                      <option key={code} value={code}>{code}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    maxLength={32}
                    onChange={(event) => setLanguage(event.target.value)}
                    placeholder="Automatic detection"
                    value={language}
                  />
                )}
              </label>
            )}
          </div>
          {capabilities?.context && (
            <label className="field">
              <span>Recognition context <em>optional</em></span>
              <textarea
                maxLength={1000}
                onChange={(event) => setInitialPrompt(event.target.value)}
                placeholder="Names, vocabulary, or context that may improve recognition."
                rows={3}
                value={initialPrompt}
              />
              <small>{initialPrompt.length} / 1000</small>
            </label>
          )}
          {selectedModel && selectedModel.load_state !== "loaded" ? (
            <button
              className="button primary full"
              onClick={() => startModelTransition(selectedModel.id, "loaded")}
              type="button"
            >
              Load {selectedModel.display_name}
            </button>
          ) : (
            <button
              className="button primary full"
              disabled={!file || busy || captureActive || selectedModel?.load_state !== "loaded"}
              type="submit"
            >
              {busy
                ? <LoaderCircle className="spin" size={19} />
                : <AudioLines size={19} />}
              {busy ? "Transcribing…" : "Transcribe audio"}
            </button>
          )}
        </form>

        <section className="panel transcript-result">
          <header>
            <div>
              <h2>{result ? "Ready to use" : "Waiting for audio"}</h2>
            </div>
            {result && (
              <button className="button outline" onClick={copyText} type="button">
                {copied ? <Check size={16} /> : <Clipboard size={16} />}
                {copied ? "Copied" : "Copy"}
              </button>
            )}
          </header>
          {result ? (
            <>
              <p className="transcript-text">{result.text}</p>
              <dl className="transcript-meta">
                <div>
                  <dt>Duration</dt>
                  <dd>{result.duration_seconds.toFixed(2)} sec</dd>
                </div>
                <div>
                  <dt>Language</dt>
                  <dd>{result.language || "Automatic"}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>{result.model}</dd>
                </div>
              </dl>
            </>
          ) : (
            <div className="transcript-placeholder">
              <AudioLines size={30} />
              <p>Your completed transcript and source metadata appear here.</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
