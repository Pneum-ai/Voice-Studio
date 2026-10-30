/** Persistent marketing-compatible light, dark, and system themes. */

import { useEffect, useMemo, useState } from "react";

const STORAGE_KEY = "pneum-voice-theme";
const VALID_MODES = new Set(["light", "dark", "system"]);

/** Resolve the operating system color preference. */
function systemTheme() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

/** Read a valid stored preference or default to system mode. */
function storedTheme() {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return VALID_MODES.has(stored) ? stored : "system";
}

/** Apply, persist, and reactively resolve the selected theme mode. */
export function useTheme() {
  const [mode, setMode] = useState(storedTheme);
  const [system, setSystem] = useState(systemTheme);
  const resolved = mode === "system" ? system : mode;

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const update = (event) => setSystem(event.matches ? "dark" : "light");
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = resolved;
    document.documentElement.style.colorScheme = resolved;
    window.localStorage.setItem(STORAGE_KEY, mode);
  }, [mode, resolved]);

  return useMemo(
    () => ({ mode, resolved, setMode }),
    [mode, resolved],
  );
}
