/** Editor project command and history tests. */

import { describe, expect, it } from "vitest";

import {
  applyProjectCommand,
  clipDuration,
  createClip,
  createHistory,
  EMPTY_PROJECT,
  historyReducer,
  projectDuration,
  snapClipStart,
} from "./project";

const SOURCE = {
  id: "source-1",
  kind: "upload",
  name: "Greeting",
  detail: "Uploaded audio",
  duration: 4,
  assetId: "source-1",
};

describe("editor project state", () => {
  it("adds a source and non-destructive clip as one command", () => {
    const clip = { ...createClip(SOURCE, 2), id: "clip-1" };
    const project = applyProjectCommand(EMPTY_PROJECT, {
      type: "add_clip",
      source: SOURCE,
      clip,
    });

    expect(project.sources).toEqual([SOURCE]);
    expect(project.clips).toEqual([clip]);
    expect(clipDuration(clip)).toBe(4);
    expect(projectDuration(project)).toBe(6);
  });

  it("undoes and redoes timeline edits", () => {
    const clip = { ...createClip(SOURCE, 0), id: "clip-1" };
    let history = createHistory();
    history = historyReducer(history, {
      type: "execute",
      command: { type: "add_clip", source: SOURCE, clip },
    });
    history = historyReducer(history, {
      type: "execute",
      command: {
        type: "update_clip",
        clipId: clip.id,
        changes: { inPoint: 1, outPoint: 3 },
      },
    });

    expect(clipDuration(history.present.clips[0])).toBe(2);
    history = historyReducer(history, { type: "undo" });
    expect(clipDuration(history.present.clips[0])).toBe(4);
    history = historyReducer(history, { type: "redo" });
    expect(clipDuration(history.present.clips[0])).toBe(2);
  });

  it("snaps moved clips to the origin and neighboring clip edges", () => {
    const first = { ...createClip(SOURCE, 0), id: "clip-1", outPoint: 2 };
    const second = { ...createClip(SOURCE, 4), id: "clip-2", outPoint: 1 };
    const third = { ...createClip(SOURCE, 5), id: "clip-3", outPoint: 2 };
    const project = {
      ...EMPTY_PROJECT,
      sources: [SOURCE],
      clips: [first, second, third],
    };

    expect(snapClipStart(project, 0.22, second.id)).toEqual({
      start: 0,
      snapped: true,
    });
    expect(snapClipStart(project, 2.24, second.id)).toEqual({
      start: 2,
      snapped: true,
    });
    expect(snapClipStart(project, 4.14, second.id)).toEqual({
      start: 4,
      snapped: true,
    });
    expect(snapClipStart(project, 2.7, second.id)).toEqual({
      start: 2.7,
      snapped: false,
    });
  });
});
