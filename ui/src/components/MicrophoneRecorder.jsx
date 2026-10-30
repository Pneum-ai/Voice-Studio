/** Short microphone capture with permission handling and explicit resource cleanup. */

import { LoaderCircle, Mic, Square } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { formatDuration, problemMessage } from "../utils/format";
import { startWavRecording } from "../utils/recording";

const MAX_SECONDS = 60;

/** Hand a completed microphone WAV to the surrounding audio workflow. */
export function MicrophoneRecorder({ disabled, onRecorded, onActiveChange }) {
  const [phase, setPhase] = useState("idle");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [level, setLevel] = useState({ peak: 0, rms: 0 });
  const [loudPeaks, setLoudPeaks] = useState(false);
  const stopRef = useRef(null);
  const sessionRef = useRef(0);

  useEffect(() => () => {
    sessionRef.current += 1;
    const stop = stopRef.current;
    stopRef.current = null;
    stop?.().catch(() => {});
  }, []);

  useEffect(() => {
    onActiveChange(phase !== "idle");
    return () => onActiveChange(false);
  }, [onActiveChange, phase]);

  const finish = useCallback(async () => {
    const stop = stopRef.current;
    if (!stop) {
      return;
    }
    const session = sessionRef.current;
    stopRef.current = null;
    setPhase("saving");
    try {
      const blob = await stop();
      if (session !== sessionRef.current) {
        return;
      }
      if (blob.size <= 44) {
        throw new Error("No audio was captured. Try recording a little longer.");
      }
      onRecorded(new File([blob], `microphone-${Date.now()}.wav`, { type: "audio/wav" }));
    } catch (failure) {
      if (session === sessionRef.current) {
        setError(problemMessage(failure));
      }
    } finally {
      if (session === sessionRef.current) {
        setPhase("idle");
      }
    }
  }, [onRecorded]);

  useEffect(() => {
    if (phase !== "recording") {
      return undefined;
    }
    const started = Date.now();
    const timer = window.setInterval(() => {
      setElapsed(Math.min(MAX_SECONDS, Math.floor((Date.now() - started) / 1000)));
    }, 250);
    const limit = window.setTimeout(finish, MAX_SECONDS * 1000);
    return () => {
      window.clearInterval(timer);
      window.clearTimeout(limit);
    };
  }, [finish, phase]);

  const start = async () => {
    const session = ++sessionRef.current;
    setError("");
    setElapsed(0);
    setLevel({ peak: 0, rms: 0 });
    setLoudPeaks(false);
    setPhase("requesting");
    try {
      const stop = await startWavRecording({ onLevel: (nextLevel) => {
        if (session !== sessionRef.current) return;
        setLevel(nextLevel);
        if (nextLevel.peak >= 0.98) setLoudPeaks(true);
      } });
      if (session !== sessionRef.current) {
        await stop();
        return;
      }
      stopRef.current = stop;
      setPhase("recording");
    } catch (failure) {
      if (session !== sessionRef.current) {
        return;
      }
      const messages = {
        NotAllowedError: "Microphone access was denied. Allow it in your browser and try again.",
        NotFoundError: "No microphone was found. Connect one and try again.",
        NotReadableError: "Microphone unavailable. Close other recording apps and try again.",
      };
      setError(messages[failure.name] || problemMessage(failure));
      setPhase("idle");
    }
  };

  const cancel = async () => {
    sessionRef.current += 1;
    const stop = stopRef.current;
    stopRef.current = null;
    setPhase("idle");
    setElapsed(0);
    setLoudPeaks(false);
    await stop?.().catch(() => {});
  };

  const recording = phase === "recording";
  const waiting = phase === "requesting" || phase === "saving";
  return (
    <div className="microphone-recorder">
      <div className="microphone-recorder-row">
        <div>
          <strong>Or record with your microphone</strong>
          <p role="status">
            {recording ? "Recording — stop when you’re ready."
              : phase === "requesting" ? "Allow microphone access in your browser."
                : phase === "saving" ? "Preparing your recording…"
                  : "Record up to 60 seconds, then preview and transcribe."}
          </p>
        </div>
        <div className="microphone-recorder-actions">
          {recording && <span aria-label="Recording duration">{formatDuration(elapsed)}</span>}
          <button
            className={`button small ${recording ? "secondary" : "outline"}`}
            disabled={disabled || waiting}
            onClick={recording ? finish : start}
            type="button"
          >
            {waiting ? <LoaderCircle className="spin" size={16} />
              : recording ? <Square size={16} /> : <Mic size={16} />}
            {recording ? "Stop recording" : waiting ? "Please wait…" : "Record microphone"}
          </button>
          {(recording || phase === "requesting") && (
            <button className="text-button" onClick={cancel} type="button">Cancel</button>
          )}
        </div>
      </div>
      {recording && (
        <div className="microphone-level">
          <label>
            Microphone level
            <meter
              aria-label="Microphone level"
              high={-6}
              low={-40}
              max={0}
              min={-60}
              optimum={-12}
              value={Math.max(-60, Math.min(0, 20 * Math.log10(level.peak || 0.000001)))}
            />
          </label>
          <span className={level.peak >= 0.98 ? "microphone-warning" : undefined}>
            {level.peak >= 0.98 ? "Too loud — lower your microphone input level."
            : level.rms < 0.01 ? "Quiet — speak normally and check the level."
              : "Good level — keep speaking naturally."}
          </span>
        </div>
      )}
      {loudPeaks && (
        <p className="microphone-warning" role="status">
          Loud peaks detected. Move farther from the mic or lower its input level.
          {" "}Listen for distortion before saving; re-record if it sounds harsh.
        </p>
      )}
      {error && <p className="microphone-error" role="alert">{error}</p>}
    </div>
  );
}
