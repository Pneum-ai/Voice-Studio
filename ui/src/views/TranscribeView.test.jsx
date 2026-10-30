/** Microphone-to-transcript flow, permission races, and recording cleanup. */

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { startWavRecording } from "../utils/recording";
import { TranscribeView } from "./TranscribeView";

vi.mock("../utils/recording", () => ({ startWavRecording: vi.fn() }));
vi.mock("../components/AudioPlayer", () => ({
  AudioPlayer: ({ title }) => <div aria-label="Audio preview">{title}</div>,
}));
vi.mock("../state/ToastContext", () => ({ useToast: () => ({ showToast: vi.fn() }) }));
vi.mock("../state/AppContext", () => ({
  useApp: () => ({
    sttModel: "whisper-base",
    setSttModel: vi.fn(),
    startModelTransition: vi.fn(),
    models: [{
      id: "whisper-base",
      kind: "transcription",
      display_name: "Whisper Base",
      load_state: "loaded",
      backend_supported: true,
      transcription: { language_mode: "optional", languages: ["en"], context: true },
    }],
  }),
}));

let stop;
beforeEach(() => {
  stop = vi.fn().mockResolvedValue(new Blob([new Uint8Array(100)], { type: "audio/wav" }));
  vi.mocked(startWavRecording).mockResolvedValue(stop);
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  vi.useRealTimers();
});

it("records, previews, and submits a WAV without requiring a file upload", async () => {
  const user = userEvent.setup();
  const transcribe = vi.spyOn(api, "transcribe").mockResolvedValue({
    text: "A microphone test.", duration_seconds: 2, language: "en", model: "whisper-base",
  });
  render(<TranscribeView />);
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  expect(screen.getByRole("button", { name: "Transcribe audio" })).toBeDisabled();
  expect(screen.getByLabelText("Choose an audio file")).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Stop recording" }));
  expect(stop).toHaveBeenCalledTimes(1);
  expect(await screen.findByLabelText("Audio preview")).toHaveTextContent(/microphone-.*\.wav/);
  expect(screen.getByLabelText("Choose an audio file")).not.toBeRequired();
  await user.click(screen.getByRole("button", { name: "Transcribe audio" }));
  expect(await screen.findByText("A microphone test.")).toBeInTheDocument();
  const [file, model] = transcribe.mock.calls[0];
  expect(file).toBeInstanceOf(File);
  expect(file.type).toBe("audio/wav");
  expect(file.size).toBe(100);
  expect(model).toBe("whisper-base");
});

it("shows a useful permission error and allows another attempt", async () => {
  vi.mocked(startWavRecording).mockRejectedValueOnce(new DOMException("Denied", "NotAllowedError"));
  const user = userEvent.setup();
  render(<TranscribeView />);
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Microphone access was denied");
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  expect(await screen.findByRole("button", { name: "Stop recording" })).toBeEnabled();
});

it("shows input levels and keeps a loud-peak warning after stopping", async () => {
  const user = userEvent.setup();
  render(<TranscribeView />);
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  const { onLevel } = startWavRecording.mock.calls[0][0];
  act(() => onLevel({ peak: 1, rms: 0.5 }));
  expect(screen.getByRole("meter", { name: "Microphone level" })).toHaveAttribute("value", "0");
  expect(screen.getByText(/Too loud/)).toBeInTheDocument();
  act(() => onLevel({ peak: 0.3, rms: 0.1 }));
  expect(screen.getByText(/Good level/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Stop recording" }));
  expect(screen.getByText(/Loud peaks detected/)).toBeInTheDocument();
});

it("discards a cancelled recording while preserving the previous upload", async () => {
  const user = userEvent.setup();
  render(<TranscribeView />);
  await user.upload(screen.getByLabelText("Choose an audio file"),
    new File(["audio"], "original.wav", { type: "audio/wav" }));
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  await user.click(screen.getByRole("button", { name: "Cancel" }));
  expect(stop).toHaveBeenCalledTimes(1);
  expect(screen.getByLabelText("Audio preview")).toHaveTextContent("original.wav");
});

it.each(["cancel", "leave"])("releases late microphone access after %s", async (action) => {
  let grant;
  vi.mocked(startWavRecording).mockReturnValue(new Promise((resolve) => { grant = resolve; }));
  const user = userEvent.setup();
  const view = render(<TranscribeView />);
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  if (action === "cancel") {
    await user.click(screen.getByRole("button", { name: "Cancel" }));
  } else {
    view.unmount();
  }
  await act(async () => grant(stop));
  expect(stop).toHaveBeenCalledTimes(1);
  expect(screen.queryByLabelText("Audio preview")).not.toBeInTheDocument();
});

it("releases an active microphone when leaving the screen", async () => {
  const user = userEvent.setup();
  const view = render(<TranscribeView />);
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  view.unmount();
  await waitFor(() => expect(stop).toHaveBeenCalledTimes(1));
});

it("automatically finishes a test clip at sixty seconds", async () => {
  vi.useFakeTimers();
  render(<TranscribeView />);
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "Record microphone" }));
  });
  await act(async () => vi.advanceTimersByTimeAsync(60_000));
  expect(stop).toHaveBeenCalledTimes(1);
  expect(screen.getByLabelText("Audio preview")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Transcribe audio" })).toBeEnabled();
});
