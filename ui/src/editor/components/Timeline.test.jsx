/** Timeline drag-offset, magnetic guide, and drop behavior tests. */

import { createEvent, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Timeline } from "./Timeline";

const SOURCES = [
  { id: "source-1", name: "First", duration: 2, peaks: [20, 40] },
  { id: "source-2", name: "Second", duration: 1, peaks: [40, 20] },
];
const CLIPS = [
  {
    id: "clip-1",
    sourceId: "source-1",
    start: 0,
    inPoint: 0,
    outPoint: 2,
    gain: 1,
    fadeIn: 0,
    fadeOut: 0,
  },
  {
    id: "clip-2",
    sourceId: "source-2",
    start: 4,
    inPoint: 0,
    outPoint: 1,
    gain: 1,
    fadeIn: 0,
    fadeOut: 0,
  },
];

/** Provide the small DataTransfer surface used by the editor. */
function dataTransfer() {
  const values = new Map();
  return {
    effectAllowed: "none",
    getData: (type) => values.get(type) || "",
    setData: (type, value) => values.set(type, value),
  };
}

/** Fire a jsdom drag event with a concrete horizontal pointer coordinate. */
function fireDrag(element, kind, clientX, transfer) {
  const event = createEvent[kind](element, { dataTransfer: transfer });
  Object.defineProperty(event, "clientX", { value: clientX });
  fireEvent(element, event);
}

/** Render a deterministic two-clip timeline and return its drag targets. */
function renderTimeline(onMoveClip) {
  render(
    <Timeline
      duration={5}
      onDropSource={vi.fn()}
      onMoveClip={onMoveClip}
      onSeek={vi.fn()}
      onSelect={vi.fn()}
      playhead={0}
      project={{ version: 1, name: "Test", sources: SOURCES, clips: CLIPS }}
      selectedId={null}
    />,
  );
  const timeline = screen.getByRole("application", { name: "Audio timeline" });
  const clip = screen.getByRole("button", {
    name: "Second, starts at 4.0 seconds",
  });
  timeline.getBoundingClientRect = () => ({ left: 100, width: 1000 });
  clip.getBoundingClientRect = () => ({ left: 100, width: 36 });
  return { clip, timeline };
}

describe("editor timeline dragging", () => {
  it("preserves grab offset and snaps a clip start to a neighboring end", () => {
    const onMoveClip = vi.fn();
    const { clip, timeline } = renderTimeline(onMoveClip);
    const transfer = dataTransfer();

    fireDrag(clip, "dragStart", 118, transfer);
    fireDrag(timeline, "dragOver", 190, transfer);
    expect(screen.getByText("Snap 2.0s")).toBeInTheDocument();
    fireDrag(timeline, "drop", 190, transfer);

    expect(onMoveClip).toHaveBeenCalledWith("clip-2", 2);
  });

  it("snaps a grabbed clip exactly to zero", () => {
    const onMoveClip = vi.fn();
    const { clip, timeline } = renderTimeline(onMoveClip);
    const transfer = dataTransfer();

    fireDrag(clip, "dragStart", 118, transfer);
    fireDrag(timeline, "dragOver", 108, transfer);
    expect(screen.getByText("Snap 0.0s")).toBeInTheDocument();
    fireDrag(timeline, "drop", 108, transfer);

    expect(onMoveClip).toHaveBeenCalledWith("clip-2", 0);
  });
});
