/** Browser entry point and stylesheet composition for Pneum.ai Voice Studio. */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/shell.css";
import "./styles/components.css";
import "./styles/audio.css";
import "./styles/editor.css";
import "./styles/editor-layout.css";
import "./styles/editor-responsive.css";
import "./styles/create-voices.css";
import "./styles/library.css";
import "./styles/modal.css";
import "./styles/system.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
