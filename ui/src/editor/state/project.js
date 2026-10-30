/** Serializable edit-decision state and undoable project commands. */

export const EMPTY_PROJECT = {
  version: 1,
  name: "Untitled voice scene",
  sources: [],
  clips: [],
};

/** Create a collision-resistant browser-local identifier. */
export function createEditorId(prefix) {
  const suffix = globalThis.crypto?.randomUUID?.()
    || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}-${suffix}`;
}

/** Return one clip's audible duration after non-destructive trimming. */
export function clipDuration(clip) {
  return Math.max(0, clip.outPoint - clip.inPoint);
}

/** Return the total occupied project duration. */
export function projectDuration(project) {
  return project.clips.reduce(
    (duration, clip) => Math.max(duration, clip.start + clipDuration(clip)),
    0,
  );
}

/** Magnetize a proposed clip start to the origin or neighboring clip edges. */
export function snapClipStart(
  project,
  rawStart,
  movingClipId = null,
  threshold = 0.34,
) {
  const movingClip = project.clips.find((clip) => clip.id === movingClipId);
  const movingDuration = movingClip ? clipDuration(movingClip) : null;
  const edges = [0];
  project.clips.forEach((clip) => {
    if (clip.id !== movingClipId) {
      edges.push(clip.start, clip.start + clipDuration(clip));
    }
  });
  const candidates = [...edges];
  if (movingDuration !== null) {
    edges.forEach((edge) => {
      if (edge >= movingDuration) {
        candidates.push(edge - movingDuration);
      }
    });
  }
  const proposed = Math.max(0, rawStart);
  const closest = candidates.reduce((best, candidate) => (
    Math.abs(candidate - proposed) < Math.abs(best - proposed)
      ? candidate
      : best
  ), candidates[0]);
  if (Math.abs(closest - proposed) <= threshold) {
    return { start: Number(closest.toFixed(3)), snapped: true };
  }
  return {
    start: Number((Math.round(proposed * 10) / 10).toFixed(1)),
    snapped: false,
  };
}

/** Apply one editor command without mutating the previous project. */
export function applyProjectCommand(project, command) {
  switch (command.type) {
    case "rename":
      return { ...project, name: command.name };
    case "add_source":
      if (project.sources.some((source) => source.id === command.source.id)) {
        return project;
      }
      return { ...project, sources: [...project.sources, command.source] };
    case "add_clip": {
      const hasSource = project.sources.some(
        (source) => source.id === command.source.id,
      );
      return {
        ...project,
        sources: hasSource
          ? project.sources
          : [...project.sources, command.source],
        clips: [...project.clips, command.clip],
      };
    }
    case "update_clip":
      return {
        ...project,
        clips: project.clips.map((clip) => (
          clip.id === command.clipId ? { ...clip, ...command.changes } : clip
        )),
      };
    case "remove_clip":
      return {
        ...project,
        clips: project.clips.filter((clip) => clip.id !== command.clipId),
      };
    default:
      return project;
  }
}

/** Initialize undo history around a project snapshot. */
export function createHistory(project = EMPTY_PROJECT) {
  return { past: [], present: project, future: [] };
}

/** Reduce commands plus undo, redo, hydration, and reset operations. */
export function historyReducer(history, action) {
  if (action.type === "hydrate" || action.type === "reset") {
    return createHistory(action.project || EMPTY_PROJECT);
  }
  if (action.type === "undo" && history.past.length) {
    return {
      past: history.past.slice(0, -1),
      present: history.past.at(-1),
      future: [history.present, ...history.future],
    };
  }
  if (action.type === "redo" && history.future.length) {
    return {
      past: [...history.past, history.present],
      present: history.future[0],
      future: history.future.slice(1),
    };
  }
  if (action.type !== "execute") {
    return history;
  }
  const next = applyProjectCommand(history.present, action.command);
  if (next === history.present) {
    return history;
  }
  return {
    past: [...history.past.slice(-49), history.present],
    present: next,
    future: [],
  };
}

/** Build a new clip from a decoded source and timeline position. */
export function createClip(source, start = 0) {
  return {
    id: createEditorId("clip"),
    sourceId: source.id,
    start: Math.max(0, start),
    inPoint: 0,
    outPoint: source.duration,
    gain: 1,
    fadeIn: 0,
    fadeOut: 0,
  };
}
