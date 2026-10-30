/** Accessible waveform trimmer for a decoded reference-audio upload. */

import { Pause, Play, Scissors } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { formatDuration } from "../utils/format";
import { summarizeWaveform } from "../utils/waveform";

const MIN_SELECTION_SECONDS = 0.25;
const WAVEFORM_BAR_COUNT = 72;

/** Render a waveform with draggable start and end boundaries. */
export function AudioTrimmer({
  audioBuffer,
  end,
  onChange,
  src,
  start,
  title,
}) {
  const audioRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(start);
  const [playing, setPlaying] = useState(false);
  const duration = audioBuffer?.duration || 0;
  const bars = useMemo(() => (
    audioBuffer
      ? summarizeWaveform(
        audioBuffer.getChannelData(0),
        WAVEFORM_BAR_COUNT,
      )
      : []
  ), [audioBuffer]);
  const startPercent = duration ? (start / duration) * 100 : 0;
  const endPercent = duration ? (end / duration) * 100 : 100;
  const playPercent = duration ? (currentTime / duration) * 100 : startPercent;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || (audio.currentTime >= start && audio.currentTime <= end)) {
      return;
    }
    audio.currentTime = start;
    setCurrentTime(start);
  }, [end, start]);

  /** Start or pause playback constrained to the selected interval. */
  const togglePlayback = async () => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    if (!audio.paused) {
      audio.pause();
      return;
    }
    if (audio.currentTime < start || audio.currentTime >= end) {
      audio.currentTime = start;
    }
    await audio.play();
  };

  /** Stop playback precisely at the trim boundary. */
  const updatePlayback = () => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    if (audio.currentTime >= end) {
      audio.pause();
      audio.currentTime = start;
      setCurrentTime(start);
      return;
    }
    setCurrentTime(audio.currentTime);
  };

  /** Move the trim start without crossing the end boundary. */
  const changeStart = (event) => {
    const value = Math.min(
      Number(event.target.value),
      end - MIN_SELECTION_SECONDS,
    );
    onChange({ end, start: Math.max(0, value) });
  };

  /** Move the trim end without crossing the start boundary. */
  const changeEnd = (event) => {
    const value = Math.max(
      Number(event.target.value),
      start + MIN_SELECTION_SECONDS,
    );
    onChange({ end: Math.min(duration, value), start });
  };

  /** Scrub the preview within its selected interval without changing the trim. */
  const seek = (event) => {
    const position = Math.max(start, Math.min(end, Number(event.target.value)));
    if (audioRef.current) audioRef.current.currentTime = position;
    setCurrentTime(position);
  };

  return (
    <div className="audio-trimmer">
      <audio
        onEnded={() => setPlaying(false)}
        onPause={() => setPlaying(false)}
        onPlay={() => setPlaying(true)}
        onTimeUpdate={updatePlayback}
        preload="metadata"
        ref={audioRef}
        src={src}
      >
        <track kind="captions" />
      </audio>
      <div className="trimmer-heading">
        <span><Scissors size={15} />Trim reference</span>
        <strong>{formatDuration(end - start)} selected</strong>
      </div>
      <div className="trimmer-main">
        <button
          aria-label={playing ? "Pause selection" : "Play selection"}
          className="audio-play trimmer-play"
          disabled={!duration}
          onClick={togglePlayback}
          type="button"
        >
          {playing ? <Pause fill="currentColor" /> : <Play fill="currentColor" />}
        </button>
        <div className="trimmer-track">
          <div
            aria-hidden="true"
            className="trimmer-waveform"
            style={{
              "--trim-end": `${endPercent}%`,
              "--trim-play": `${playPercent}%`,
              "--trim-start": `${startPercent}%`,
            }}
          >
            {bars.map((height, index) => (
              <span
                className={
                  index / bars.length >= start / duration
                  && index / bars.length <= end / duration
                    ? "is-selected"
                    : ""
                }
                key={`${index}-${height}`}
                style={{ height: `${height}%` }}
              />
            ))}
            <i className="trim-selection" />
            <i className="trim-playhead" />
          </div>
          <input
            aria-label="Trim start"
            className="trim-range trim-range-start"
            disabled={!duration}
            max={duration}
            min="0"
            onChange={changeStart}
            step="0.01"
            type="range"
            value={start}
          />
          <input
            aria-label="Trim end"
            className="trim-range trim-range-end"
            disabled={!duration}
            max={duration}
            min="0"
            onChange={changeEnd}
            step="0.01"
            type="range"
            value={end}
          />
          <div className="trimmer-times">
            <span>Start {formatDuration(start)}</span>
            <span>End {formatDuration(end)}</span>
          </div>
        </div>
      </div>
      <label className="field">
        <span>Preview position · {formatDuration(currentTime)}</span>
        <input
          aria-label="Seek within reference sample"
          disabled={!duration}
          max={end}
          min={start}
          onChange={seek}
          step="0.01"
          type="range"
          value={Math.max(start, Math.min(end, currentTime))}
        />
      </label>
      <p>
        Scrub to review the audio, or drag either edge and re-transcribe the selected speech.
        The saved reference will contain only this interval.
      </p>
      <span className="sr-only">{title}</span>
    </div>
  );
}
