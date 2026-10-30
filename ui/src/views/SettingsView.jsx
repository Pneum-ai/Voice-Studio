/** Appearance, health, cache, and API resource settings workspace. */

import {
  Activity,
  BookOpen,
  Braces,
  CheckCircle2,
  Cpu,
  Database,
  ExternalLink,
  HeartPulse,
  RefreshCw,
  Server,
  Trash2,
} from "lucide-react";
import { useState } from "react";

import { api } from "../api/client";
import { ThemeSelector } from "../components/ThemeSelector";
import { useApp } from "../state/AppContext";
import { useToast } from "../state/ToastContext";
import { problemMessage } from "../utils/format";

/** Render local runtime status and safe maintenance controls. */
export function SettingsView({ theme }) {
  const { health, refreshHealth } = useApp();
  const { showToast } = useToast();
  const [clearing, setClearing] = useState(false);

  /** Clear derived prompt artifacts without deleting source audio. */
  const clearCache = async () => {
    setClearing(true);
    try {
      const response = await api.clearProfilePrompts();
      showToast(
        `Cleared ${response.deleted_files} derived prompt files.`,
        "success",
      );
    } catch (error) {
      showToast(problemMessage(error), "error");
    } finally {
      setClearing(false);
    }
  };

  /** Refresh service health and report the result. */
  const refresh = async () => {
    try {
      await refreshHealth();
      showToast("System status refreshed.", "success");
    } catch (error) {
      showToast(problemMessage(error), "error");
    }
  };

  return (
    <div className="settings-page">
      <section className="page-title-row">
        <div>
          <h1>Settings</h1>
          <p>Appearance, runtime health, cache maintenance, and API resources.</p>
        </div>
        <button className="button outline" onClick={refresh} type="button">
          <RefreshCw size={16} />
          Refresh status
        </button>
      </section>

      <div className="settings-grid">
        <section className="panel settings-card">
          <header>
            <span className="settings-icon"><Activity size={20} /></span>
            <div>
              <h2>System status</h2>
              <p>Current local service and inference backend.</p>
            </div>
          </header>
          <div className={`health-hero${health?.status === "ready" ? " is-ready" : ""}`}>
            <CheckCircle2 size={24} />
            <div>
              <strong>{health?.status === "ready" ? "System ready" : "Unavailable"}</strong>
              <span>Pneum.ai Voice Studio is connected to the API.</span>
            </div>
          </div>
          <dl className="settings-list">
            <div>
              <dt><Server size={16} />Version</dt>
              <dd>{health?.version || "—"}</dd>
            </div>
            <div>
              <dt><Cpu size={16} />Backend</dt>
              <dd>
                {health?.backend || "—"}
                {health?.runtime_version && ` · ${health.runtime_version}`}
              </dd>
            </div>
            <div>
              <dt><HeartPulse size={16} />Device</dt>
              <dd>
                {health?.device?.toUpperCase() || "—"}
                {health?.device_name && ` · ${health.device_name}`}
              </dd>
            </div>
            <div>
              <dt><BookOpen size={16} />Bundled UI</dt>
              <dd>{health?.ui_enabled ? "Enabled" : "Development proxy"}</dd>
            </div>
          </dl>
          {health?.device_advisory && (
            <p className="runtime-advisory">{health.device_advisory}</p>
          )}
        </section>

        <section className="panel settings-card">
          <header>
            <span className="settings-icon"><Braces size={20} /></span>
            <div>
              <h2>Appearance</h2>
              <p>Match the operating system or choose a fixed theme.</p>
            </div>
          </header>
          <div className="theme-setting">
            <span>Theme mode</span>
            <ThemeSelector mode={theme.mode} onChange={theme.setMode} />
          </div>
          <div className="theme-preview-row">
            <button
              aria-pressed={theme.mode === "light"}
              className="theme-preview theme-preview-light"
              onClick={() => theme.setMode("light")}
              type="button"
            >
              <span />
              <strong>Light</strong>
              <small>Clean blue-white surfaces</small>
            </button>
            <button
              aria-pressed={theme.mode === "dark"}
              className="theme-preview theme-preview-dark"
              onClick={() => theme.setMode("dark")}
              type="button"
            >
              <span />
              <strong>Dark</strong>
              <small>Deep navy studio surfaces</small>
            </button>
          </div>
        </section>

        <section className="panel settings-card">
          <header>
            <span className="settings-icon"><Database size={20} /></span>
            <div>
              <h2>Prompt cache</h2>
              <p>Clear safe derived Qwen prompt tensors and combined references.</p>
            </div>
          </header>
          <p className="settings-copy">
            Voice names, source samples, transcripts, and saved generations are
            preserved. Prompts are rebuilt the next time each voice generates.
          </p>
          <button
            className="button danger"
            disabled={clearing}
            onClick={clearCache}
            type="button"
          >
            <Trash2 size={16} />
            {clearing ? "Clearing…" : "Clear derived prompts"}
          </button>
        </section>

        <section className="panel settings-card">
          <header>
            <span className="settings-icon"><ExternalLink size={20} /></span>
            <div>
              <h2>API resources</h2>
              <p>Inspect and integrate with the same API used by this UI.</p>
            </div>
          </header>
          <div className="resource-links">
            <a href="/docs" rel="noreferrer" target="_blank">
              <BookOpen size={17} />
              Interactive API docs
              <ExternalLink size={14} />
            </a>
            <a href="/openapi.json" rel="noreferrer" target="_blank">
              <Braces size={17} />
              OpenAPI JSON
              <ExternalLink size={14} />
            </a>
            <a href="/health" rel="noreferrer" target="_blank">
              <HeartPulse size={17} />
              Raw health response
              <ExternalLink size={14} />
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}
