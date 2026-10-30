/** Branded responsive application shell and primary navigation. */

import {
  AudioLines,
  Box,
  FolderOpen,
  Music,
  Mic2,
  PenLine,
  Settings,
  SlidersHorizontal,
} from "lucide-react";

import { ThemeSelector } from "./ThemeSelector";

const NAV_ITEMS = [
  { path: "/", label: "Create", Icon: PenLine },
  { path: "/voices", label: "Voices", Icon: Mic2 },
  { path: "/library", label: "Library", Icon: FolderOpen },
  { path: "/editor", label: "Editor", Icon: Music },
  { path: "/transcribe", label: "Transcribe", Icon: AudioLines },
  { path: "/models", label: "Models", Icon: Box },
];

const MOBILE_NAV_ITEMS = [
  ...NAV_ITEMS,
  { path: "/settings", label: "Settings", Icon: Settings },
];

/** Render the shared Pneum.ai sidebar, top bar, and mobile navigation. */
export function AppShell({
  children,
  health,
  loading,
  navigate,
  path,
  theme,
}) {
  const ready = health?.status === "ready";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => navigate("/")} type="button">
          <img alt="" src="/brand-mark.png" />
          <span>Pneum<span>.ai</span></span>
        </button>
        <nav aria-label="Primary navigation">
          {NAV_ITEMS.map(({ path: itemPath, label, Icon }) => (
            <button
              aria-current={path === itemPath ? "page" : undefined}
              className={path === itemPath ? "is-active" : ""}
              key={itemPath}
              onClick={() => navigate(itemPath)}
              type="button"
            >
              <Icon size={21} strokeWidth={1.7} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <button
          aria-current={path === "/settings" ? "page" : undefined}
          className={`sidebar-settings${path === "/settings" ? " is-active" : ""}`}
          onClick={() => navigate("/settings")}
          type="button"
        >
          <Settings size={21} strokeWidth={1.7} />
          <span>Settings</span>
        </button>
      </aside>

      <div className="app-main">
        <header className="topbar">
          <div className={`system-status${ready ? " is-ready" : ""}`}>
            <span />
            {loading ? "Connecting…" : ready ? "System ready" : "Needs attention"}
          </div>
          <ThemeSelector mode={theme.mode} onChange={theme.setMode} />
          <button
            aria-label="Open settings"
            className="icon-button top-settings"
            onClick={() => navigate("/settings")}
            type="button"
          >
            <SlidersHorizontal size={18} />
          </button>
        </header>
        <main>{children}</main>
      </div>

      <nav aria-label="Mobile navigation" className="mobile-nav">
        {MOBILE_NAV_ITEMS.map(({ path: itemPath, label, Icon }) => (
          <button
            aria-label={label}
            aria-current={path === itemPath ? "page" : undefined}
            className={path === itemPath ? "is-active" : ""}
            key={itemPath}
            onClick={() => navigate(itemPath)}
            type="button"
          >
            <Icon size={20} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
