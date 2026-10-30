/** Model-job SSE behavior across supervised worker restarts. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { followJob } from "./jobs";

class FakeEventSource {
  /** Keep the most recent fake source available to each test. */
  static latest = null;

  /** Capture listeners and expose a close spy. */
  constructor(url) {
    this.url = url;
    this.listeners = new Map();
    this.close = vi.fn();
    this.onerror = null;
    FakeEventSource.latest = this;
  }

  /** Register one named server-sent event listener. */
  addEventListener(name, listener) {
    this.listeners.set(name, listener);
  }

  /** Deliver one JSON lifecycle snapshot. */
  emit(name, job) {
    this.listeners.get(name)?.({ data: JSON.stringify(job) });
  }
}

describe("followJob", () => {
  beforeEach(() => {
    vi.stubGlobal("EventSource", FakeEventSource);
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    FakeEventSource.latest = null;
  });

  it("leaves EventSource retry active while the supervised worker is offline", async () => {
    fetch.mockRejectedValue(new TypeError("connection refused"));
    const onUpdate = vi.fn();
    const onError = vi.fn();

    followJob("switch-job", onUpdate, onError);
    const source = FakeEventSource.latest;
    await source.onerror();

    expect(source.url).toBe("/v1/jobs/switch-job/events");
    expect(source.close).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();

    source.emit("complete", { id: "switch-job", state: "complete" });
    expect(onUpdate).toHaveBeenCalledWith({ id: "switch-job", state: "complete" });
    expect(source.close).toHaveBeenCalledOnce();
  });

  it("stops retrying when a healthy replacement server does not know the job", async () => {
    fetch.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: "Not Found",
      json: async () => ({ detail: "Job was not found." }),
    });
    const onError = vi.fn();

    followJob("missing-job", vi.fn(), onError);
    const source = FakeEventSource.latest;
    await source.onerror();

    expect(onError).toHaveBeenCalledOnce();
    expect(source.close).toHaveBeenCalledOnce();
  });
});
