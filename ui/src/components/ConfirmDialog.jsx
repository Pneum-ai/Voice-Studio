/** Confirmation modal for destructive profile, sample, and audio actions. */

import { AlertTriangle } from "lucide-react";

import { Modal } from "./Modal";

/** Render a focused destructive-action confirmation dialog. */
export function ConfirmDialog({
  title,
  description,
  confirmLabel = "Delete",
  busy = false,
  open,
  onClose,
  onConfirm,
}) {
  return (
    <Modal
      description={description}
      onClose={onClose}
      open={open}
      title={title}
    >
      <div className="confirm-visual">
        <AlertTriangle size={22} />
        <span>This action cannot be undone.</span>
      </div>
      <footer className="modal-actions">
        <button className="button secondary" onClick={onClose} type="button">
          Cancel
        </button>
        <button
          className="button danger"
          disabled={busy}
          onClick={onConfirm}
          type="button"
        >
          {busy ? "Working…" : confirmLabel}
        </button>
      </footer>
    </Modal>
  );
}
