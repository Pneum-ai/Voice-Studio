/** Lightweight history-based routing for the bundled single-page app. */

import { useEffect, useState } from "react";

const ROUTES = new Set([
  "/",
  "/voices",
  "/library",
  "/editor",
  "/transcribe",
  "/models",
  "/settings",
]);

/** Normalize the current browser path into a supported studio route. */
function normalizedPath() {
  return ROUTES.has(window.location.pathname) ? window.location.pathname : "/";
}

/** Track routes and expose navigation without a router dependency. */
export function useRoute() {
  const [path, setPath] = useState(normalizedPath);

  useEffect(() => {
    const handlePopState = () => setPath(normalizedPath());
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  /** Push a normalized route and update local navigation state. */
  const navigate = (nextPath) => {
    if (nextPath === path) {
      return;
    }
    window.history.pushState({}, "", nextPath);
    setPath(nextPath);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return { path, navigate };
}
