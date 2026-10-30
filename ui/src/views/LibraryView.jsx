/** Searchable playback library for stored speech and exact input metadata. */

import {
  BookOpen,
  CalendarClock,
  ChevronRight,
  Filter,
  Search,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import { AudioPlayer } from "../components/AudioPlayer";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { EmptyState } from "../components/EmptyState";
import { Modal } from "../components/Modal";
import { useApp } from "../state/AppContext";
import { useToast } from "../state/ToastContext";
import { formatDate, problemMessage } from "../utils/format";

const EMPTY_FILTERS = {
  query: "",
  profile_id: "",
  model: "",
};

/** Render generation filters, cards, details, download, and deletion. */
export function LibraryView() {
  const { models, profiles } = useApp();
  const { showToast } = useToast();
  const [draftFilters, setDraftFilters] = useState(EMPTY_FILTERS);
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [items, setItems] = useState([]);
  const [nextCursor, setNextCursor] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);

  /** Load one filtered cursor page and optionally append it. */
  const load = useCallback(async (cursor = null, append = false) => {
    setLoading(true);
    try {
      const response = await api.listGenerations({
        ...filters,
        cursor,
        limit: 25,
      });
      setItems((current) => append ? [...current, ...response.items] : response.items);
      setNextCursor(response.next_cursor);
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setLoading(false);
    }
  }, [filters, showToast]);

  useEffect(() => {
    load();
  }, [load]);

  /** Apply the current draft filters from the first page. */
  const applyFilters = (event) => {
    event.preventDefault();
    setFilters({ ...draftFilters });
  };

  /** Reset every library filter to its unfiltered state. */
  const clearFilters = () => {
    setDraftFilters(EMPTY_FILTERS);
    setFilters(EMPTY_FILTERS);
  };

  /** Fetch complete stored metadata before opening the detail modal. */
  const openDetail = async (generationId) => {
    setDetailLoading(true);
    setDetail({ id: generationId });
    try {
      setDetail(await api.getGeneration(generationId));
    } catch (error) {
      setDetail(null);
      showToast(problemMessage(error), "error");
    } finally {
      setDetailLoading(false);
    }
  };

  /** Delete the confirmed saved audio and refresh the first page. */
  const deleteGeneration = async () => {
    if (!deleteTarget) {
      return;
    }
    setDeleting(true);
    try {
      await api.deleteGeneration(deleteTarget.id);
      setDeleteTarget(null);
      if (detail?.id === deleteTarget.id) {
        setDetail(null);
      }
      await load();
      showToast("Saved generation deleted.", "success");
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="library-page">
      <section className="page-title-row">
        <div>
          <h1>Library</h1>
          <p>Search, listen to, inspect, download, and remove generated speech.</p>
        </div>
        <span className="count-badge">{items.length} loaded</span>
      </section>

      <form className="panel library-filters" onSubmit={applyFilters}>
        <label className="search-field">
          <Search size={17} />
          <span className="sr-only">Search input text</span>
          <input
            onChange={(event) => setDraftFilters((current) => ({
              ...current,
              query: event.target.value,
            }))}
            placeholder="Search exact input text"
            value={draftFilters.query}
          />
        </label>
        <label className="field compact-field">
          <span className="sr-only">Filter by voice</span>
          <select
            onChange={(event) => setDraftFilters((current) => ({
              ...current,
              profile_id: event.target.value,
            }))}
            value={draftFilters.profile_id}
          >
            <option value="">All voices</option>
            {profiles.map((profile) => (
              <option key={profile.id} value={profile.id}>{profile.name}</option>
            ))}
          </select>
        </label>
        <label className="field compact-field">
          <span className="sr-only">Filter by model</span>
          <select
            onChange={(event) => setDraftFilters((current) => ({
              ...current,
              model: event.target.value,
            }))}
            value={draftFilters.model}
          >
            <option value="">All models</option>
            {models.filter((model) => model.kind === "tts").map((model) => (
              <option key={model.id} value={model.id}>{model.display_name}</option>
            ))}
          </select>
        </label>
        <button className="button primary" type="submit">
          <Filter size={16} />
          Apply
        </button>
        <button className="button text" onClick={clearFilters} type="button">
          Clear
        </button>
      </form>

      {!loading && !items.length ? (
        <EmptyState
          action={filters.query || filters.profile_id || filters.model
            ? (
              <button className="button outline" onClick={clearFilters} type="button">
                Clear filters
              </button>
            )
            : null}
          description={filters.query || filters.profile_id || filters.model
            ? "Try a broader search or clear the active filters."
            : "Enable Save to library when generating speech to keep it here."}
          icon={BookOpen}
          title={filters.query || filters.profile_id || filters.model
            ? "No matching generations"
            : "Your library is empty"}
        />
      ) : (
        <div className="generation-grid">
          {items.map((generation) => (
            <article className="panel generation-card" key={generation.id}>
              <header>
                <div className="generation-meta">
                  <strong>{generation.profile_name}</strong>
                  <span>{generation.model}</span>
                </div>
                <button
                  aria-label="Delete generation"
                  className="icon-button danger-text"
                  onClick={() => setDeleteTarget(generation)}
                  type="button"
                >
                  <Trash2 size={16} />
                </button>
              </header>
              <p className="generation-text">{generation.input_text}</p>
              {generation.instructions && (
                <p className="generation-instructions">
                  “{generation.instructions}”
                </p>
              )}
              <AudioPlayer
                compact
                downloadName={`${generation.profile_name}-${generation.id}.wav`}
                src={generation.audio_url}
                title={generation.input_text}
              />
              <footer>
                <span>
                  <CalendarClock size={14} />
                  {formatDate(generation.created_at)}
                </span>
                <button
                  className="text-button"
                  onClick={() => openDetail(generation.id)}
                  type="button"
                >
                  Details
                  <ChevronRight size={15} />
                </button>
              </footer>
            </article>
          ))}
        </div>
      )}

      {nextCursor && (
        <button
          className="button outline load-more"
          disabled={loading}
          onClick={() => load(nextCursor, true)}
          type="button"
        >
          {loading ? "Loading…" : "Load more"}
        </button>
      )}

      <Modal
        onClose={() => setDetail(null)}
        open={Boolean(detail)}
        title="Generation details"
      >
        {detailLoading || !detail?.input_text ? (
          <p className="modal-loading">Loading generation…</p>
        ) : (
          <div className="generation-detail">
            <AudioPlayer
              downloadName={`${detail.profile_name}-${detail.id}.wav`}
              src={detail.audio_url}
              title={detail.input_text}
            />
            <dl>
              <div><dt>Voice</dt><dd>{detail.profile_name}</dd></div>
              <div><dt>Model</dt><dd>{detail.model}</dd></div>
              <div><dt>Language</dt><dd>{detail.language}</dd></div>
              <div><dt>Seed</dt><dd>{detail.seed ?? "Random"}</dd></div>
              <div>
                <dt>Duration</dt>
                <dd>{detail.duration_seconds.toFixed(2)} seconds</dd>
              </div>
              <div><dt>Sample rate</dt><dd>{detail.sample_rate} Hz</dd></div>
              <div><dt>Created</dt><dd>{formatDate(detail.created_at)}</dd></div>
            </dl>
            <section>
              <h3>Input text</h3>
              <p>{detail.input_text}</p>
            </section>
            {detail.instructions && (
              <section>
                <h3>Instructions</h3>
                <p>{detail.instructions}</p>
              </section>
            )}
          </div>
        )}
      </Modal>

      <ConfirmDialog
        busy={deleting}
        description="The generated WAV and its stored metadata will be removed."
        onClose={() => setDeleteTarget(null)}
        onConfirm={deleteGeneration}
        open={Boolean(deleteTarget)}
        title="Delete this generation?"
      />
    </div>
  );
}
