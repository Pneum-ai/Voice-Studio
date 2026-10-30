/** Audio source shelf for saved, uploaded, and recorded clips. */

import {
  AudioLines,
  CirclePlus,
  ChevronLeft,
  ChevronRight,
  FileUp,
  Library,
  Mic,
  Search,
  Square,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";

/** Format a short editor duration label. */
function formatDuration(seconds) {
  const safe = Number.isFinite(seconds) ? seconds : 0;
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${String(Math.floor(safe % 60)).padStart(2, "0")}`;
}

/** Render available audio and source-capture actions. */
export function SourceLibrary({
  busy,
  collapsed,
  items,
  loading,
  onAdd,
  onRecord,
  onToggle,
  onUpload,
  recording,
}) {
  const inputRef = useRef(null);
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => items.filter((item) => (
    item.name.toLowerCase().includes(query.toLowerCase())
  )), [items, query]);

  /** Forward a drag payload understood by the timeline. */
  const startDrag = (event, sourceId) => {
    event.dataTransfer.effectAllowed = "copy";
    event.dataTransfer.setData("application/x-pneum-source", sourceId);
  };

  return (
    <aside className={`panel editor-sources${collapsed ? " is-collapsed" : ""}`}>
      <header>
        <div>
          <h2>Clips</h2>
        </div>
        <span className="editor-source-count">{items.length}</span>
        <button
          aria-label={collapsed ? "Open source audio" : "Collapse source audio"}
          className="icon-button editor-source-toggle"
          onClick={onToggle}
          type="button"
        >
          {collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}
        </button>
      </header>
      {!collapsed && <>
        <div className="editor-capture-actions">
        <button
          className="button outline small"
          disabled={busy || recording}
          onClick={() => inputRef.current?.click()}
          type="button"
        >
          <FileUp size={15} />
          Upload
        </button>
        <button
          className={`button small ${recording ? "danger" : "outline"}`}
          disabled={busy}
          onClick={onRecord}
          type="button"
        >
          {recording ? <Square size={14} /> : <Mic size={15} />}
          {recording ? "Stop" : "Record"}
        </button>
        <input
          accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
          className="sr-only"
          multiple
          onChange={(event) => {
            onUpload([...event.target.files]);
            event.target.value = "";
          }}
          ref={inputRef}
          type="file"
        />
        </div>
        <label className="search-field editor-source-search">
        <Search size={15} />
        <span className="sr-only">Search source audio</span>
        <input
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search clips"
          value={query}
        />
        </label>
        <p className="editor-source-hint">
          Drag a clip to the timeline or add at the playhead.
        </p>
        <div className="editor-source-list">
        {filtered.map((item) => (
          <article
            className="editor-source-card"
            draggable
            key={item.id}
            onDragStart={(event) => startDrag(event, item.id)}
          >
            <span className={`editor-source-icon is-${item.kind}`}>
              {item.kind === "generation" ? <Library size={16} /> : <AudioLines size={16} />}
            </span>
            <div>
              <strong title={item.name}>{item.name}</strong>
              <small>{item.detail} · {formatDuration(item.duration)}</small>
            </div>
            <button
              aria-label={`Add ${item.name} at playhead`}
              className="icon-button"
              disabled={busy}
              onClick={() => onAdd(item.id)}
              type="button"
            >
              <CirclePlus size={16} />
            </button>
          </article>
        ))}
        {!loading && !filtered.length && (
          <div className="editor-source-empty">
            <AudioLines size={22} />
            <span>{query ? "No matching audio" : "Upload or record your first clip"}</span>
          </div>
        )}
        {loading && <div className="editor-source-empty">Loading library…</div>}
        </div>
      </>}
    </aside>
  );
}
