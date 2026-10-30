/** Server-sent event helpers for model lifecycle progress. */

import { api } from "./client";

const JOB_EVENTS = ["queued", "running", "complete", "failed", "cancelled"];
const TERMINAL_STATES = new Set(["complete", "failed", "cancelled"]);
const RECONNECT_TIMEOUT_MS = 5 * 60 * 1000;

/** Follow one model job until it reaches a terminal state. */
export function followJob(jobId, onUpdate, onError) {
  const source = new EventSource(`/v1/jobs/${jobId}/events`);
  let reconnectStartedAt = null;
  let closed = false;

  /** Stop browser reconnection and suppress later callbacks. */
  const close = () => {
    closed = true;
    source.close();
  };

  /** Decode and forward one named job event. */
  const handleEvent = (event) => {
    try {
      const job = JSON.parse(event.data);
      reconnectStartedAt = null;
      onUpdate(job);
      if (TERMINAL_STATES.has(job.state)) {
        close();
      }
    } catch (error) {
      onError?.(error);
      close();
    }
  };

  JOB_EVENTS.forEach((name) => source.addEventListener(name, handleEvent));
  source.onerror = async () => {
    if (closed) {
      return;
    }
    reconnectStartedAt ??= Date.now();
    try {
      const job = await api.getJob(jobId);
      onUpdate(job);
      if (TERMINAL_STATES.has(job.state)) {
        close();
      }
    } catch (error) {
      if (error?.status === 404) {
        onError?.(error);
        close();
      }
    }
    if (!closed && Date.now() - reconnectStartedAt >= RECONNECT_TIMEOUT_MS) {
      onError?.(new Error("Model job did not reconnect after the server restart."));
      close();
    }
  };

  return close;
}
