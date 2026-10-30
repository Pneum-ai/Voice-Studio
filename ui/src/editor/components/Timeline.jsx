/** Single-track, horizontally scrollable voice-scene timeline. */

import { GripVertical, Music2 } from "lucide-react";
import { useState } from "react";

import { clipDuration, snapClipStart } from "../state/project";

const PIXELS_PER_SECOND = 36;

/** Render one clip waveform from its persisted peak cache. */
function ClipWaveform({ peaks }) {
  const bars = peaks?.length ? peaks : Array.from({ length: 32 }, () => 24);
  return (
    <span aria-hidden="true" className="editor-clip-waveform">
      {bars.map((height, index) => (
        <i key={`${index}-${height}`} style={{ height: `${height}%` }} />
      ))}
    </span>
  );
}

/** Render the timeline ruler, drop lane, clips, and playhead. */
export function Timeline({
  duration,
  onDropSource,
  onMoveClip,
  onSeek,
  onSelect,
  playhead,
  project,
  selectedId,
}) {
  const [dragState, setDragState] = useState(null);
  const [dropPreview, setDropPreview] = useState(null);
  const timelineDuration = Math.max(30, Math.ceil(duration + 5));
  const width = timelineDuration * PIXELS_PER_SECOND;
  const sources = new Map(project.sources.map((source) => [source.id, source]));

  /** Convert a pointer location within the full lane to seconds. */
  const eventTime = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return Math.max(0, (event.clientX - rect.left) / PIXELS_PER_SECOND);
  };

  /** Resolve drag identity, grab offset, and magnetic landing position. */
  const dropPosition = (event) => {
    const clipId = event.dataTransfer.getData("application/x-pneum-clip")
      || dragState?.clipId
      || "";
    const sourceId = event.dataTransfer.getData("application/x-pneum-source");
    const storedOffset = event.dataTransfer.getData(
      "application/x-pneum-grab-offset",
    );
    const grabOffset = storedOffset
      ? Number(storedOffset)
      : dragState?.grabOffset || 0;
    const result = snapClipStart(
      project,
      eventTime(event) - grabOffset,
      clipId || null,
    );
    return { ...result, clipId, sourceId };
  };

  /** Accept source copies and clip moves from native drag-and-drop. */
  const drop = (event) => {
    event.preventDefault();
    const { clipId, sourceId, start } = dropPosition(event);
    setDragState(null);
    setDropPreview(null);
    if (sourceId) {
      onDropSource(sourceId, start);
    } else if (clipId) {
      onMoveClip(clipId, start);
    }
  };

  return (
    <section className="panel editor-timeline-panel">
      <header>
        <div>
          <h2>Scene timeline</h2>
        </div>
        <span>Single track · {duration.toFixed(1)} sec</span>
      </header>
      <div className="editor-timeline-scroll">
        <div
          aria-label="Audio timeline"
          className="editor-timeline"
          onClick={(event) => onSeek(Math.min(duration, eventTime(event)))}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget)) {
              setDropPreview(null);
            }
          }}
          onDragOver={(event) => {
            event.preventDefault();
            const result = dropPosition(event);
            setDropPreview({ start: result.start, snapped: result.snapped });
          }}
          onDrop={drop}
          role="application"
          style={{ width }}
        >
          <div className="editor-ruler">
            {Array.from({ length: timelineDuration + 1 }, (_, second) => (
              <span key={second} style={{ left: second * PIXELS_PER_SECOND }}>
                {second % 5 === 0 ? `${second}s` : ""}
              </span>
            ))}
          </div>
          <div className="editor-track-label"><Music2 size={15} /> Voice scene</div>
          <div className="editor-track-lane">
            {!project.clips.length && (
              <p>Drop audio here to begin your scene</p>
            )}
            {project.clips.map((clip) => {
              const source = sources.get(clip.sourceId);
              return (
                <button
                  aria-label={[
                    source?.name || "Audio clip",
                    `starts at ${clip.start.toFixed(1)} seconds`,
                  ].join(", ")}
                  aria-pressed={selectedId === clip.id}
                  className={`editor-clip${selectedId === clip.id ? " is-selected" : ""}`}
                  draggable
                  key={clip.id}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelect(clip.id);
                  }}
                  onDragStart={(event) => {
                    event.dataTransfer.effectAllowed = "move";
                    event.dataTransfer.setData("application/x-pneum-clip", clip.id);
                    const rect = event.currentTarget.getBoundingClientRect();
                    const grabOffset = (event.clientX - rect.left) / PIXELS_PER_SECOND;
                    setDragState({
                      clipId: clip.id,
                      grabOffset: Math.max(0, grabOffset),
                    });
                    event.dataTransfer.setData(
                      "application/x-pneum-grab-offset",
                      String(Math.max(0, grabOffset)),
                    );
                  }}
                  onDragEnd={() => {
                    setDragState(null);
                    setDropPreview(null);
                  }}
                  style={{
                    left: clip.start * PIXELS_PER_SECOND,
                    width: Math.max(54, clipDuration(clip) * PIXELS_PER_SECOND),
                  }}
                  type="button"
                >
                  <GripVertical size={14} />
                  <span className="editor-clip-copy">
                    <strong>{source?.name || "Audio"}</strong>
                    <small>{clipDuration(clip).toFixed(1)} sec</small>
                  </span>
                  <ClipWaveform peaks={source?.peaks} />
                </button>
              );
            })}
          </div>
          <span
            aria-hidden="true"
            className="editor-playhead"
            style={{ left: playhead * PIXELS_PER_SECOND }}
          />
          {dropPreview && (
            <span
              aria-hidden="true"
              className={`editor-drop-guide${dropPreview.snapped ? " is-snapped" : ""}`}
              style={{ left: dropPreview.start * PIXELS_PER_SECOND }}
            >
              <b>{dropPreview.snapped ? "Snap" : "Place"} {dropPreview.start.toFixed(1)}s</b>
            </span>
          )}
        </div>
      </div>
    </section>
  );
}
