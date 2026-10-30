/** Create and rename forms for minimal name-only voice profiles. */

import { useEffect, useState } from "react";

import { Modal } from "./Modal";

/** Render a controlled profile-name dialog. */
export function ProfileFormModal({
  initialName = "",
  open,
  title,
  submitLabel,
  onClose,
  onSubmit,
}) {
  const [name, setName] = useState(initialName);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (open) {
      setName(initialName);
      setBusy(false);
    }
  }, [initialName, open]);

  /** Validate and submit the trimmed voice name. */
  const submit = async (event) => {
    event.preventDefault();
    if (!name.trim()) {
      return;
    }
    setBusy(true);
    try {
      await onSubmit(name.trim());
      onClose();
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      description="Profiles contain a name and one or more reference samples."
      onClose={onClose}
      open={open}
      title={title}
    >
      <form className="modal-form" onSubmit={submit}>
        <label className="field">
          <span>Voice name</span>
          <input
            autoFocus
            maxLength={100}
            onChange={(event) => setName(event.target.value)}
            placeholder="Example voice"
            required
            value={name}
          />
        </label>
        <footer className="modal-actions">
          <button className="button secondary" onClick={onClose} type="button">
            Cancel
          </button>
          <button className="button primary" disabled={busy} type="submit">
            {busy ? "Saving…" : submitLabel}
          </button>
        </footer>
      </form>
    </Modal>
  );
}
