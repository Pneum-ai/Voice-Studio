/** Model download byte-progress rendering tests. */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ToastProvider } from "../state/ToastContext";
import { ModelsView } from "./ModelsView";

const appState = vi.hoisted(() => ({
  jobs: [{
    id: "download-job",
    kind: "model_state",
    resource_id: "qwen3-asr-0.6B",
    state: "running",
    progress: 0.47,
    stage: "downloading",
    detail: "812 MB of 1.8 GB · 18 MB/s · 58s remaining",
    created_at: "2026-08-24T00:00:00Z",
    updated_at: "2026-08-24T00:01:00Z",
  }],
  models: [{
    id: "qwen3-asr-0.6B",
    display_name: "Qwen3-ASR 0.6B",
    kind: "transcription",
    family: "Qwen3-ASR",
    provider: "Qwen",
    transcription: { streaming: true },
    download_state: "absent",
    load_state: "unloaded",
    size_bytes: 812_000_000,
    backend_supported: true,
    native_streaming: false,
    active_job_id: "download-job",
  }],
  refreshJobs: vi.fn(),
  refreshModels: vi.fn(),
  startModelTransition: vi.fn(),
}));

vi.mock("../state/AppContext", () => ({
  useApp: () => appState,
}));

describe("ModelsView", () => {
  it("shows byte progress, rate, ETA, and proportional completion", () => {
    render(
      <ToastProvider>
        <ModelsView />
      </ToastProvider>,
    );

    expect(screen.getAllByText(appState.jobs[0].detail).length).toBeGreaterThan(0);
    expect(screen.getByText("47%")).toBeInTheDocument();
    expect(
      screen.getByRole("progressbar", { name: "Qwen3-ASR 0.6B progress" }),
    ).toHaveValue(0.47);
  });
});
