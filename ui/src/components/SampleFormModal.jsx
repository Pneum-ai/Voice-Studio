/** Reference-sample upload, microphone capture, and transcript editing modal. */

import {
  FileAudio,
  LoaderCircle,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { decodeAudioFile } from "../utils/audio_edit";
import { problemMessage } from "../utils/format";
import { AudioTrimmer } from "./AudioTrimmer";
import { Modal } from "./Modal";
import { MicrophoneRecorder } from "./MicrophoneRecorder";

/** Render the add or edit workflow for one voice sample. */
export function SampleFormModal({
  initialTranscript = "",
  mode = "add",
  open,
  onClose,
  onLoadTranscriptionModel,
  onSubmit,
  onTranscribe,
  transcriptionModel,
  transcriptionReady,
}) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [transcript, setTranscript] = useState(initialTranscript);
  const [busy, setBusy] = useState(false);
  const [captureActive, setCaptureActive] = useState(false);
  const transcriptionRequest = useRef(0);
  const [transcribing, setTranscribing] = useState(false);
  const [transcriptionError, setTranscriptionError] = useState("");
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [decoding, setDecoding] = useState(false);
  const [decodeError, setDecodeError] = useState("");
  const [selection, setSelection] = useState({ end: 0, start: 0 });
  const [selectionNeedsReview, setSelectionNeedsReview] = useState(false);
  const [transcriptionLanguage, setTranscriptionLanguage] = useState("");
  const transcriptionCapabilities = transcriptionModel?.transcription;
  const languages = transcriptionCapabilities?.languages || [];
  const defaultLanguage = languages.includes("en") ? "en" : languages[0] || "";

  useEffect(() => {
    setFile(null);
    setPreviewUrl("");
    setAudioBuffer(null);
    setDecoding(false);
    if (open) {
      setTranscript(initialTranscript);
      setBusy(false);
      setTranscribing(false);
      setTranscriptionError("");
      setDecodeError("");
      setSelection({ end: 0, start: 0 });
      setSelectionNeedsReview(false);
      setCaptureActive(false);
    }
    return () => { transcriptionRequest.current += 1; };
  }, [initialTranscript, open]);

  useEffect(() => {
    setTranscriptionLanguage(defaultLanguage);
  }, [defaultLanguage, open]);

  useEffect(() => {
    if (!open || !file) {
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file, open]);

  useEffect(() => {
    if (!open || !file) {
      return undefined;
    }
    let current = true;
    setDecoding(true);
    setDecodeError("");
    decodeAudioFile(file)
      .then((decoded) => {
        if (!current) {
          return;
        }
        setAudioBuffer(decoded);
        setSelection({ end: decoded.duration, start: 0 });
      })
      .catch((error) => {
        if (current) {
          setDecodeError(problemMessage(error));
        }
      })
      .finally(() => {
        if (current) {
          setDecoding(false);
        }
      });
    return () => {
      current = false;
    };
  }, [file, open]);

  /** Return lightweight trim coordinates without copying decoded PCM audio. */
  const selectedTrim = () => {
    if (!audioBuffer) {
      return null;
    }
    return {
      endSeconds: selection.end,
      startSeconds: selection.start,
    };
  };

  /** Transcribe the selected sample and place editable text in the field. */
  const transcribeFile = async (
    selectedFile,
    trim = null,
    ready = transcriptionReady,
  ) => {
    if (!selectedFile || !ready) {
      setTranscriptionError(
        `${transcriptionModel?.display_name || "The transcription model"} must be loaded first.`,
      );
      return;
    }
    setTranscribing(true);
    setTranscriptionError("");
    const request = ++transcriptionRequest.current;
    try {
      const result = transcriptionCapabilities?.language_mode === "required"
        ? await onTranscribe(selectedFile, trim, transcriptionLanguage)
        : await onTranscribe(selectedFile, trim);
      if (request === transcriptionRequest.current) {
        setTranscript(result.text);
        setSelectionNeedsReview(false);
      }
    } catch (error) {
      if (request === transcriptionRequest.current) setTranscriptionError(problemMessage(error));
    } finally {
      if (request === transcriptionRequest.current) setTranscribing(false);
    }
  };

  /** Preview and transcribe an uploaded file or completed microphone recording. */
  const selectFile = (selectedFile) => {
    if (!selectedFile) return;
    transcriptionRequest.current += 1;
    setFile(selectedFile);
    setAudioBuffer(null);
    setPreviewUrl("");
    setSelection({ end: 0, start: 0 });
    setSelectionNeedsReview(false);
    setTranscript("");
    setTranscriptionError("");
    if (selectedFile) {
      transcribeFile(selectedFile, null);
    }
  };

  /** Explicitly load the model, then resume the pending transcription. */
  const loadAndTranscribe = async () => {
    if (!file) {
      return;
    }
    setTranscribing(true);
    setTranscriptionError("");
    const request = ++transcriptionRequest.current;
    try {
      await onLoadTranscriptionModel();
      if (request !== transcriptionRequest.current) return;
      await transcribeFile(file, selectedTrim(), true);
    } catch (error) {
      if (request === transcriptionRequest.current) {
        setTranscriptionError(problemMessage(error));
        setTranscribing(false);
      }
    }
  };

  /** Re-run recognition using only the currently selected audio interval. */
  const retranscribeSelection = async () => {
    await transcribeFile(file, selectedTrim());
  };

  /** Update trim boundaries and flag the current transcript for review. */
  const changeSelection = (nextSelection) => {
    setSelection(nextSelection);
    setSelectionNeedsReview(true);
  };

  /** Keep manual transcript edits as an explicit review of the selection. */
  const changeTranscript = (event) => {
    setTranscript(event.target.value);
    setSelectionNeedsReview(false);
  };

  /** Validate and submit sample audio with its exact transcript. */
  const submit = async (event) => {
    event.preventDefault();
    if (captureActive || busy || transcribing || !transcript.trim()
      || (mode === "add" && (!file || !audioBuffer))) {
      return;
    }
    setBusy(true);
    try {
      await onSubmit({
        file,
        transcript: transcript.trim(),
        trim: mode === "add" ? selectedTrim() : null,
      });
      onClose();
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      description={mode === "add"
        ? "Upload or record clear speech, then preview, trim, and verify the transcript."
        : "Keep the transcript exact and describe how the stored audio sounds."}
      onClose={onClose}
      open={open}
      title={mode === "add" ? "Add reference sample" : "Edit sample"}
    >
      <form className="modal-form" onSubmit={submit}>
        {mode === "add" && (
          <label className={`file-drop${file ? " has-file" : ""}`}>
            <FileAudio size={24} />
            <strong>{file ? file.name : "Choose an audio file"}</strong>
            <span>WAV, FLAC, MP3, M4A, and other common audio formats</span>
            <input
              accept="audio/*"
              disabled={busy || transcribing || captureActive}
              onChange={(event) => {
                selectFile(event.target.files?.[0]);
                event.target.value = "";
              }}
              required={!file}
              type="file"
            />
          </label>
        )}
        {mode === "add" && (
          <MicrophoneRecorder
            disabled={busy || transcribing || decoding}
            onActiveChange={setCaptureActive}
            onRecorded={selectFile}
          />
        )}
        {mode === "add" && previewUrl && !captureActive && (
          <div className="sample-upload-preview">
            <div className="sample-preview-heading">
              <span>
                <Sparkles size={15} />
                {transcribing
                  ? `Transcribing with ${transcriptionModel?.display_name || "the model"}…`
                  : "Review audio and transcript"}
              </span>
              {!transcriptionReady && !transcribing && (
                <button
                  className="button outline small"
                  onClick={loadAndTranscribe}
                  type="button"
                >
                  Load & transcribe
                </button>
              )}
            </div>
            {decoding && (
              <p className="trimmer-status">
                <LoaderCircle className="spin" size={15} />
                Preparing waveform and trim controls…
              </p>
            )}
            {audioBuffer && (
              <AudioTrimmer
                audioBuffer={audioBuffer}
                end={selection.end}
                onChange={changeSelection}
                src={previewUrl}
                start={selection.start}
                title={file?.name || "Reference sample"}
              />
            )}
            {decodeError && (
              <p className="inline-error">{decodeError}</p>
            )}
            {transcriptionCapabilities?.language_mode === "required" && (
              <label className="field">
                <span>Sample language</span>
                <select
                  onChange={(event) => setTranscriptionLanguage(event.target.value)}
                  value={transcriptionLanguage}
                >
                  {transcriptionCapabilities.languages.map((code) => (
                    <option key={code} value={code}>{code}</option>
                  ))}
                </select>
              </label>
            )}
            <div className="sample-transcribe-actions">
              <button
                className="button outline small"
                disabled={
                  decoding
                  || transcribing
                  || !audioBuffer
                  || !transcriptionReady
                }
                onClick={retranscribeSelection}
                type="button"
              >
                <RefreshCw size={15} />
                Re-transcribe selection
              </button>
              {selectionNeedsReview && (
                <small>Selection changed. Re-transcribe or edit the text.</small>
              )}
            </div>
            {transcriptionError && (
              <p className="inline-error">{transcriptionError}</p>
            )}
          </div>
        )}
        <label className="field">
          <span>
            Exact transcript
            {transcribing && <LoaderCircle className="spin" size={15} />}
          </span>
          <textarea
            autoFocus={mode === "edit"}
            disabled={transcribing || captureActive || busy}
            maxLength={5000}
            onChange={changeTranscript}
            placeholder="Type exactly what is spoken in the sample."
            required
            rows={5}
            value={transcript}
          />
          <small>{transcript.length} / 5000</small>
        </label>
        <footer className="modal-actions">
          <button className="button secondary" onClick={onClose} type="button">
            Cancel
          </button>
          <button
            className="button primary"
            disabled={
              busy
              || captureActive
              || transcribing
              || !transcript.trim()
              || (mode === "add" && !audioBuffer)
              || (mode === "add" && !file)
            }
            type="submit"
          >
            {busy ? "Saving…" : mode === "add" ? "Add sample" : "Save sample"}
          </button>
        </footer>
      </form>
    </Modal>
  );
}
