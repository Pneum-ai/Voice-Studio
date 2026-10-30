/** Application navigation and theme persistence integration tests. */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

/** Build one successful JSON fetch mock response. */
function jsonResponse(payload) {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

/** Return deterministic initial resources for application tests. */
function mockApi(url) {
  if (url === "/health") {
    return jsonResponse({
      status: "ready",
      version: "0.1.0",
      backend: "test",
      device: "cpu",
      ui_enabled: true,
    });
  }
  if (url === "/v1/profiles" || url === "/v1/models" || url === "/v1/jobs") {
    return jsonResponse({ items: [] });
  }
  if (String(url).startsWith("/v1/generations")) {
    return jsonResponse({ items: [], next_cursor: null });
  }
  return jsonResponse({});
}

describe("Pneum.ai Voice Studio application", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/");
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => mockApi(url));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it("loads the API and navigates through the complete studio", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("Start with a voice")).toBeInTheDocument();
    expect(await screen.findByText("System ready")).toBeInTheDocument();
    const navigation = within(
      screen.getByRole("navigation", { name: "Primary navigation" }),
    );

    await user.click(navigation.getByRole("button", { name: "Voices" }));
    expect(
      await screen.findByRole("heading", { name: "Voices" }),
    ).toBeInTheDocument();

    await user.click(navigation.getByRole("button", { name: "Library" }));
    expect(
      await screen.findByRole("heading", { name: "Library" }),
    ).toBeInTheDocument();

    await user.click(navigation.getByRole("button", { name: "Editor" }));
    expect(
      await screen.findByRole("heading", { name: "Scene timeline" }),
    ).toBeInTheDocument();

    await user.click(navigation.getByRole("button", { name: "Transcribe" }));
    expect(
      await screen.findByRole("heading", { name: "Transcribe audio" }),
    ).toBeInTheDocument();

    await user.click(navigation.getByRole("button", { name: "Models" }));
    expect(
      await screen.findByRole("heading", { name: "Models" }),
    ).toBeInTheDocument();
  });

  it("persists the selected Pneum theme", async () => {
    const user = userEvent.setup();
    render(<App />);
    const dark = await screen.findByRole("button", { name: /Dark/i });
    await user.click(dark);
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(window.localStorage.getItem("pneum-voice-theme")).toBe("dark");
  });
});
