/** Presentation-safe formatters for audio, dates, bytes, and API problems. */

/** Format seconds as a compact media duration. */
export function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) {
    return "0:00";
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

/** Format byte counts using human-readable binary units. */
export function formatBytes(bytes) {
  if (bytes === null || bytes === undefined) {
    return "Not downloaded";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

/** Format an API timestamp using the browser locale. */
export function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

/** Extract the most useful user-facing message from an API error. */
export function problemMessage(error) {
  return error?.problem?.detail || error?.message || "Something went wrong.";
}
