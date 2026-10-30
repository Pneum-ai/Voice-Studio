/** Automatic sample transcription and audio preview integration tests. */

import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";

import { decodeAudioFile } from "../utils/audio_edit";
import { startWavRecording } from "../utils/recording";
import { SampleFormModal } from "./SampleFormModal";

vi.mock("../utils/audio_edit", () => ({
  decodeAudioFile: vi.fn(),
}));
vi.mock("../utils/recording", () => ({ startWavRecording: vi.fn() }));

beforeEach(() => {
  vi.clearAllMocks();
  decodeAudioFile.mockResolvedValue({
    duration: 10,
    getChannelData: () => new Float32Array(100),
    length: 100,
    numberOfChannels: 1,
    sampleRate: 10,
  });
});

it("records, scrubs, trims, and saves a reference without an uploaded file", async () => {
  const user = userEvent.setup();
  const stop = vi.fn().mockResolvedValue(new Blob([new Uint8Array(100)], { type: "audio/wav" }));
  startWavRecording.mockResolvedValue(stop);
  const onSubmit = vi.fn().mockResolvedValue();
  const onTranscribe = vi.fn().mockResolvedValue({ text: "Recorded reference words." });
  render(<SampleFormModal open onClose={vi.fn()} onSubmit={onSubmit}
    onTranscribe={onTranscribe} transcriptionReady />);
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  expect(screen.getByRole("button", { name: "Add sample" })).toBeDisabled();
  expect(screen.getByLabelText(/Choose an audio file/i)).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Stop recording" }));
  const scrubber = await screen.findByRole("slider", { name: "Seek within reference sample" });
  fireEvent.change(scrubber, { target: { value: "4" } });
  const audio = screen.getByRole("dialog").querySelector("audio");
  expect(audio.currentTime).toBe(4);
  expect(screen.getByRole("slider", { name: "Trim start" })).toHaveValue("0");
  fireEvent.change(screen.getByRole("slider", { name: "Trim start" }),
    { target: { value: "2" } });
  fireEvent.change(screen.getByRole("slider", { name: "Trim end" }),
    { target: { value: "8" } });
  await user.click(screen.getByRole("button", { name: "Re-transcribe selection" }));
  const file = onTranscribe.mock.calls[0][0];
  expect(file).toBeInstanceOf(File);
  expect(file.type).toBe("audio/wav");
  expect(file.size).toBe(100);
  expect(onTranscribe).toHaveBeenLastCalledWith(file, { startSeconds: 2, endSeconds: 8 });
  expect(screen.getByLabelText(/microphone-.*\.wav/i, { selector: "input" })).not.toBeRequired();
  await user.click(screen.getByRole("button", { name: "Add sample" }));
  await waitFor(() => expect(onSubmit).toHaveBeenCalledWith({
    file, transcript: "Recorded reference words.", trim: { startSeconds: 2, endSeconds: 8 },
  }));
  expect(stop).toHaveBeenCalledTimes(1);
});

it("releases the microphone when the reference dialog closes", async () => {
  const user = userEvent.setup();
  const stop = vi.fn().mockResolvedValue(new Blob([new Uint8Array(100)]));
  startWavRecording.mockResolvedValue(stop);
  const props = { onClose: vi.fn(), onSubmit: vi.fn(), onTranscribe: vi.fn() };
  const view = render(<SampleFormModal {...props} open />);
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  view.rerender(<SampleFormModal {...props} open={false} />);
  await waitFor(() => expect(stop).toHaveBeenCalledTimes(1));
  expect(props.onTranscribe).not.toHaveBeenCalled();
  expect(props.onSubmit).not.toHaveBeenCalled();
});

it("ignores a previous recording's transcript after closing and reopening", async () => {
  const user = userEvent.setup();
  let finishTranscription;
  startWavRecording.mockResolvedValue(vi.fn().mockResolvedValue(
    new Blob([new Uint8Array(100)], { type: "audio/wav" }),
  ));
  const props = {
    onClose: vi.fn(), onSubmit: vi.fn(), transcriptionReady: true,
    onTranscribe: vi.fn().mockReturnValue(new Promise((resolve) => {
      finishTranscription = resolve;
    })),
  };
  const view = render(<SampleFormModal {...props} open />);
  await user.click(screen.getByRole("button", { name: "Record microphone" }));
  await user.click(screen.getByRole("button", { name: "Stop recording" }));
  await waitFor(() => expect(props.onTranscribe).toHaveBeenCalled());
  view.rerender(<SampleFormModal {...props} open={false} />);
  view.rerender(<SampleFormModal {...props} open />);
  await act(async () => finishTranscription({ text: "Stale words." }));
  expect(screen.getByRole("textbox", { name: /Exact transcript/ })).toHaveValue("");
  expect(screen.queryByRole("slider", { name: "Seek within reference sample" })).toBeNull();
});

it("saves the exact transcript without style controls", async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn().mockResolvedValue(undefined);
  render(
    <SampleFormModal
      initialTranscript="Exact sample words."
      mode="edit"
      onClose={vi.fn()}
      onSubmit={onSubmit}
      open
    />,
  );

  expect(screen.queryByText("Style controls")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Save sample" }));
  expect(onSubmit).toHaveBeenCalledWith({
    file: null,
    transcript: "Exact sample words.",
    trim: null,
  });
});

it("trims and re-transcribes only the selected reference interval", async () => {
  const user = userEvent.setup();
  const onTranscribe = vi.fn().mockResolvedValue({
    text: "Automatically transcribed sample.",
  });
  const onSubmit = vi.fn().mockResolvedValue(undefined);
  render(
    <SampleFormModal
      onClose={vi.fn()}
      onLoadTranscriptionModel={vi.fn()}
      onSubmit={onSubmit}
      onTranscribe={onTranscribe}
      open
      transcriptionModel={{
        display_name: "Whisper Large v3 Turbo",
      }}
      transcriptionReady
    />,
  );

  const audio = new File(["sample"], "reference.wav", { type: "audio/wav" });
  await user.upload(screen.getByLabelText(/Choose an audio file/i), audio);

  expect(onTranscribe).toHaveBeenCalledWith(audio, null);
  expect(await screen.findByDisplayValue(
    "Automatically transcribed sample.",
  )).toBeInTheDocument();
  expect(screen.queryByText("Style controls")).not.toBeInTheDocument();
  const start = await screen.findByRole("slider", { name: "Trim start" });
  expect(screen.getByRole("slider", { name: "Trim end" })).toHaveValue("10");

  fireEvent.change(start, { target: { value: "2" } });
  expect(screen.getByText(/Selection changed/i)).toBeInTheDocument();
  await user.click(screen.getByRole("button", {
    name: "Re-transcribe selection",
  }));

  expect(onTranscribe).toHaveBeenLastCalledWith(
    audio,
    { endSeconds: 10, startSeconds: 2 },
  );
  const submit = screen.getByRole("button", { name: "Add sample" });
  await waitFor(() => expect(submit).toBeEnabled());
  fireEvent.submit(submit.closest("form"));
  await waitFor(() => expect(onSubmit).toHaveBeenCalled());
  expect(onSubmit).toHaveBeenCalledWith({
    file: audio,
    transcript: "Automatically transcribed sample.",
    trim: { endSeconds: 10, startSeconds: 2 },
  });
});
