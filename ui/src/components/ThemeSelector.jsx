/** Shared light, dark, and system appearance selector. */

import { Monitor, Moon, Sun } from "lucide-react";

const OPTIONS = [
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
  { value: "system", label: "System", Icon: Monitor },
];

/** Render the compact marketing-style theme segmented control. */
export function ThemeSelector({ mode, onChange, compact = false }) {
  return (
    <div className={`theme-selector${compact ? " theme-selector-compact" : ""}`}>
      {OPTIONS.map(({ value, label, Icon }) => (
        <button
          aria-pressed={mode === value}
          className={mode === value ? "is-active" : ""}
          key={value}
          onClick={() => onChange(value)}
          type="button"
        >
          <Icon aria-hidden="true" size={15} strokeWidth={1.8} />
          {!compact && <span>{label}</span>}
        </button>
      ))}
    </div>
  );
}
