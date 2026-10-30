/** Root application composition, lazy routes, and fatal error recovery. */

import { Component, lazy, Suspense } from "react";

import { AppShell } from "./components/AppShell";
import { useRoute } from "./hooks/useRoute";
import { useTheme } from "./hooks/useTheme";
import { AppProvider, useApp } from "./state/AppContext";
import { ToastProvider } from "./state/ToastContext";
import { CreateView } from "./views/CreateView";

const LibraryView = lazy(() =>
  import("./views/LibraryView").then((module) => ({
    default: module.LibraryView,
  })),
);
const EditorView = lazy(() =>
  import("./views/EditorView").then((module) => ({
    default: module.EditorView,
  })),
);
const ModelsView = lazy(() =>
  import("./views/ModelsView").then((module) => ({
    default: module.ModelsView,
  })),
);
const SettingsView = lazy(() =>
  import("./views/SettingsView").then((module) => ({
    default: module.SettingsView,
  })),
);
const TranscribeView = lazy(() =>
  import("./views/TranscribeView").then((module) => ({
    default: module.TranscribeView,
  })),
);
const VoicesView = lazy(() =>
  import("./views/VoicesView").then((module) => ({
    default: module.VoicesView,
  })),
);

const VIEW_MAP = {
  "/voices": VoicesView,
  "/library": LibraryView,
  "/editor": EditorView,
  "/transcribe": TranscribeView,
  "/models": ModelsView,
};

/** Replace unrecoverable render failures with a branded restart action. */
class ErrorBoundary extends Component {
  /** Initialize empty error state for the child application tree. */
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  /** Capture the thrown render error for the fallback interface. */
  static getDerivedStateFromError(error) {
    return { error };
  }

  /** Render the application or its fatal recovery state. */
  render() {
    if (this.state.error) {
      return (
        <main className="fatal-error">
          <img alt="" src="/brand-mark.png" />
          <h1>The interface hit an unexpected error.</h1>
          <p>{this.state.error.message}</p>
          <button
            className="button primary"
            onClick={() => window.location.reload()}
            type="button"
          >
            Reload application
          </button>
        </main>
      );
    }
    return this.props.children;
  }
}

/** Bind shared state, theme, routing, and the selected lazy workspace. */
function Application() {
  const route = useRoute();
  const theme = useTheme();
  const { health, loading } = useApp();
  const View = VIEW_MAP[route.path] || CreateView;

  return (
    <AppShell
      health={health}
      loading={loading}
      navigate={route.navigate}
      path={route.path}
      theme={theme}
    >
      <Suspense
        fallback={
          <div className="view-loading" role="status">
            <span />
            Loading workspace…
          </div>
        }
      >
        {route.path === "/settings" ? (
          <SettingsView theme={theme} />
        ) : (
          <View navigate={route.navigate} />
        )}
      </Suspense>
    </AppShell>
  );
}

/** Provide global error, toast, and API state boundaries. */
export default function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <AppProvider>
          <Application />
        </AppProvider>
      </ToastProvider>
    </ErrorBoundary>
  );
}
