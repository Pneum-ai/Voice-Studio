/** Editor playback, history, draft, and export controls. */

import {
  Download,
  Pause,
  Play,
  Redo2,
  RotateCcw,
  Trash2,
  Undo2,
} from "lucide-react";

/** Format a timeline time as minutes and seconds. */
function formatTime(seconds) {
  const safe = Math.max(0, seconds || 0);
  return `${Math.floor(safe / 60)}:${String(Math.floor(safe % 60)).padStart(2, "0")}`;
}

/** Render transport and project-level actions. */
export function Transport({
  busy,
  canRedo,
  canUndo,
  duration,
  onClear,
  onExport,
  onPlayPause,
  onRedo,
  onRestart,
  onUndo,
  playing,
  playhead,
}) {
  return (
    <div className="panel editor-transport">
      <div className="editor-play-controls">
        <button
          aria-label="Restart scene"
          className="icon-button"
          disabled={!duration}
          onClick={onRestart}
          type="button"
        >
          <RotateCcw size={17} />
        </button>
        <button
          className="button primary editor-play-button"
          disabled={busy || !duration}
          onClick={onPlayPause}
          type="button"
        >
          {playing ? <Pause size={17} /> : <Play size={17} />}
          {playing ? "Pause" : "Play"}
        </button>
        <output>{formatTime(playhead)} / {formatTime(duration)}</output>
      </div>
      <div className="editor-history-controls">
        <button
          aria-label="Undo"
          className="icon-button"
          disabled={!canUndo || busy}
          onClick={onUndo}
          type="button"
        >
          <Undo2 size={17} />
        </button>
        <button
          aria-label="Redo"
          className="icon-button"
          disabled={!canRedo || busy}
          onClick={onRedo}
          type="button"
        >
          <Redo2 size={17} />
        </button>
        <button
          aria-label="Clear project"
          className="icon-button danger-text"
          disabled={busy || !duration}
          onClick={onClear}
          type="button"
        >
          <Trash2 size={17} />
        </button>
        <button
          className="button outline"
          disabled={busy || !duration}
          onClick={onExport}
          type="button"
        >
          <Download size={16} />
          {busy ? "Preparing…" : "Export WAV"}
        </button>
      </div>
    </div>
  );
}
