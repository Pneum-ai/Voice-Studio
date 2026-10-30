/** Model switching keeps cloned references separate from preset speakers. */
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { useApp } from "../state/AppContext";
import { CreateView } from "./CreateView";

vi.mock("../state/AppContext", () => ({ useApp: vi.fn() }));
vi.mock("../state/ToastContext", () => ({ useToast: () => ({ showToast: vi.fn() }) }));
vi.mock("../components/AudioPlayer", () => ({ AudioPlayer: () => <div>Audio ready</div> }));

const models = [
  { id: "base", display_name: "Base", kind: "tts", backend_supported: true,
    load_state: "loaded", speech: { voice_mode: "clone", instructions: false, speakers: [] } },
  { id: "custom", display_name: "CustomVoice", kind: "tts", backend_supported: true,
    load_state: "loaded", speech: { voice_mode: "builtin", instructions: true,
      speakers: ["ryan", "serena"] } },
];

/** Supply real state transitions so controls and submitted payloads can diverge if broken. */
function Harness({ profiles = [], availableModels = models }) {
  const [ttsModel, setTtsModel] = useState("base");
  useApp.mockReturnValue({ models: availableModels, profiles, ttsModel, setTtsModel,
    selectedProfileId: profiles[0]?.id || "", setSelectedProfileId: vi.fn() });
  return <CreateView navigate={vi.fn()} />;
}

afterEach(() => vi.restoreAllMocks());

describe("speech model selection", () => {
  it("opens with the loaded model despite an old selection, then allows a manual switch", () => {
    render(<Harness availableModels={[
      { ...models[0], load_state: "unloaded" }, models[1],
    ]} />);
    expect(screen.getByLabelText("Qwen model")).toHaveValue("custom");
    expect(screen.getByLabelText("Built-in speaker")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Qwen model"), { target: { value: "base" } });
    expect(screen.getByLabelText("Qwen model")).toHaveValue("base");
    expect(screen.queryByLabelText("Built-in speaker")).not.toBeInTheDocument();
  });

  it("saves a built-in speaker from an empty library", async () => {
    const save = vi.spyOn(api, "createGeneration").mockResolvedValue({ input_text: "Hello." });
    render(<Harness />);
    fireEvent.change(screen.getByLabelText("Qwen model"), { target: { value: "custom" } });
    expect(screen.queryByText("Manage samples")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Built-in speaker"), { target: { value: "ryan" } });
    fireEvent.change(screen.getByLabelText(/^Text/), { target: { value: "Hello." } });
    fireEvent.change(screen.getByLabelText(/^Instructions/), {
      target: { value: "Speak softly." },
    });
    fireEvent.click(screen.getByLabelText("Save to library"));
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));
    await waitFor(() => expect(save).toHaveBeenCalledOnce());
    expect(save.mock.calls[0][0]).toMatchObject({ speaker: "ryan", instructions: "Speak softly." });
    expect(save.mock.calls[0][0]).not.toHaveProperty("profile_id");
  });

  it("clears preset instructions on switching back to cloning", async () => {
    const save = vi.spyOn(api, "createGeneration").mockResolvedValue({ input_text: "Hello." });
    render(<Harness profiles={[{ id: "clone", name: "My voice", sample_count: 1 }]} />);
    expect(screen.queryByLabelText(/^Instructions/)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Qwen model"), { target: { value: "custom" } });
    expect(within(screen.getByLabelText("Built-in speaker")).queryByText(/My voice/)).toBeNull();
    fireEvent.change(screen.getByLabelText(/^Instructions/), { target: { value: "Whisper." } });
    fireEvent.change(screen.getByLabelText("Qwen model"), { target: { value: "base" } });
    fireEvent.change(screen.getByLabelText(/^Text/), { target: { value: "Hello." } });
    fireEvent.click(screen.getByLabelText("Save to library"));
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));
    await waitFor(() => expect(save).toHaveBeenCalledOnce());
    expect(save.mock.calls[0][0]).toMatchObject({ profile_id: "clone", instructions: null });
    expect(save.mock.calls[0][0]).not.toHaveProperty("speaker");
  });
});
