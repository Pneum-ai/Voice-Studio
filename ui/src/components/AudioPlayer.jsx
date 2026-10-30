/** Accessible audio playback, seeking, volume, and download controls. */

import { Download, Pause, Play, Volume2, VolumeX } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { formatDuration } from "../utils/format";
import { fallbackWaveformBars, loadWaveformBars } from "../utils/waveform";

/** Render one native-audio-backed branded player. */
export function AudioPlayer({
  src,
  title = "Generated speech",
  downloadName = "speech.wav",
  compact = false,
  autoPlay = false,
}) {
  const playerRef = useRef(null);
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(0.85);
  const barCount = compact ? 52 : 84;
  const [bars, setBars] = useState(() => fallbackWaveformBars(barCount));
  const progress = duration ? currentTime / duration : 0;

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setBars(fallbackWaveformBars(barCount));
    const controller = new AbortController();
    const load = () => {
      loadWaveformBars(src, barCount, controller.signal)
        .then(setBars)
        .catch((error) => {
          if (error.name !== "AbortError") {
            setBars(fallbackWaveformBars(barCount));
          }
        });
    };
    let observer = null;
    if ("IntersectionObserver" in window && playerRef.current) {
      observer = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          load();
        }
      }, { rootMargin: "160px" });
      observer.observe(playerRef.current);
    } else {
      load();
    }
    return () => {
      controller.abort();
      observer?.disconnect();
    };
  }, [barCount, src]);

  /** Toggle native playback after the initiating user gesture. */
  const toggle = async () => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    if (audio.paused) {
      await audio.play();
    } else {
      audio.pause();
    }
  };

  /** Seek the player from the waveform range control. */
  const seek = (event) => {
    const audio = audioRef.current;
    if (!audio || !duration) {
      return;
    }
    audio.currentTime = Number(event.target.value);
  };

  /** Update the native element and controlled volume slider. */
  const updateVolume = (event) => {
    const nextVolume = Number(event.target.value);
    setVolume(nextVolume);
    if (audioRef.current) {
      audioRef.current.volume = nextVolume;
    }
  };

  return (
    <div
      className={`audio-player${compact ? " audio-player-compact" : ""}`}
      ref={playerRef}
    >
      <audio
        autoPlay={autoPlay}
        onDurationChange={(event) => setDuration(event.currentTarget.duration)}
        onEnded={() => setPlaying(false)}
        onPause={() => setPlaying(false)}
        onPlay={() => setPlaying(true)}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        preload="metadata"
        ref={audioRef}
        src={src}
      />
      <button
        aria-label={playing ? `Pause ${title}` : `Play ${title}`}
        className="audio-play"
        onClick={toggle}
        type="button"
      >
        {playing ? <Pause fill="currentColor" /> : <Play fill="currentColor" />}
      </button>
      <div className="audio-track">
        <div className="waveform">
          {bars.map((height, index) => (
            <span
              className={index / bars.length <= progress ? "is-played" : ""}
              key={`${height}-${index}`}
              style={{ height: `${height}%` }}
            />
          ))}
          <i style={{ left: `${progress * 100}%` }} />
          <input
            aria-label={`Seek within ${title}`}
            max={duration || 0}
            min="0"
            onChange={seek}
            step="0.01"
            type="range"
            value={Math.min(currentTime, duration || 0)}
          />
        </div>
        <div className="audio-times">
          <span>{formatDuration(currentTime)}</span>
          <span>{formatDuration(duration)}</span>
        </div>
      </div>
      {!compact && (
        <label className="volume-control">
          {volume === 0 ? <VolumeX size={18} /> : <Volume2 size={18} />}
          <span className="sr-only">Volume</span>
          <input
            aria-label="Volume"
            max="1"
            min="0"
            onChange={updateVolume}
            step="0.05"
            type="range"
            value={volume}
          />
        </label>
      )}
      <a
        aria-label={`Download ${title}`}
        className="icon-button audio-download"
        download={downloadName}
        href={src}
      >
        <Download size={19} />
      </a>
    </div>
  );
}
