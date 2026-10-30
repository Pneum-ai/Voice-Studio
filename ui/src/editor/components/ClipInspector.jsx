/** Numeric trim, placement, gain, and fade controls for one selected clip. */

import { Trash2 } from "lucide-react";

import { clipDuration } from "../state/project";

/** Clamp one numeric edit into a safe range. */
function bounded(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, Number(value) || 0));
}

/** Render non-destructive controls for the selected clip. */
export function ClipInspector({ clip, onRemove, onUpdate, source }) {
  if (!clip || !source) {
    return (
      <aside className="panel editor-inspector is-empty">
        <h2>Select a clip</h2>
        <p>Choose timeline audio to adjust trim, placement, gain, and fades.</p>
      </aside>
    );
  }
  const duration = clipDuration(clip);
  const maxFade = duration / 2;
  return (
    <aside className="panel editor-inspector">
      <header>
        <div>
          <h2 title={source.name}>{source.name}</h2>
        </div>
        <button
          aria-label="Remove selected clip"
          className="icon-button danger-text"
          onClick={onRemove}
          type="button"
        >
          <Trash2 size={16} />
        </button>
      </header>
      <div className="editor-inspector-grid">
        <label className="field">
          <span>Timeline start <em>seconds</em></span>
          <input
            min="0"
            onChange={(event) => onUpdate({
              start: bounded(event.target.value, 0, 3600),
            })}
            step="0.1"
            type="number"
            value={clip.start.toFixed(1)}
          />
        </label>
        <label className="field">
          <span>Clip in <em>seconds</em></span>
          <input
            max={clip.outPoint - 0.05}
            min="0"
            onChange={(event) => onUpdate({
              inPoint: bounded(event.target.value, 0, clip.outPoint - 0.05),
            })}
            step="0.05"
            type="number"
            value={clip.inPoint.toFixed(2)}
          />
        </label>
        <label className="field">
          <span>Clip out <em>seconds</em></span>
          <input
            max={source.duration}
            min={clip.inPoint + 0.05}
            onChange={(event) => onUpdate({
              outPoint: bounded(
                event.target.value,
                clip.inPoint + 0.05,
                source.duration,
              ),
            })}
            step="0.05"
            type="number"
            value={clip.outPoint.toFixed(2)}
          />
        </label>
      </div>
      <div className="editor-range-grid">
        <label className="field editor-range-field">
          <span>Gain <em>{Math.round(clip.gain * 100)}%</em></span>
          <input
            max="1.5"
            min="0"
            onChange={(event) => onUpdate({ gain: Number(event.target.value) })}
            step="0.01"
            type="range"
            value={clip.gain}
          />
        </label>
        <label className="field editor-range-field">
          <span>Fade in <em>{clip.fadeIn.toFixed(1)} sec</em></span>
          <input
            max={maxFade}
            min="0"
            onChange={(event) => onUpdate({ fadeIn: Number(event.target.value) })}
            step="0.05"
            type="range"
            value={Math.min(clip.fadeIn, maxFade)}
          />
        </label>
        <label className="field editor-range-field">
          <span>Fade out <em>{clip.fadeOut.toFixed(1)} sec</em></span>
          <input
            max={maxFade}
            min="0"
            onChange={(event) => onUpdate({ fadeOut: Number(event.target.value) })}
            step="0.05"
            type="range"
            value={Math.min(clip.fadeOut, maxFade)}
          />
        </label>
      </div>
      <p className="editor-inspector-note">
        Source audio stays untouched. This clip is {duration.toFixed(2)} seconds long.
      </p>
    </aside>
  );
}
