/** Reusable empty-state presentation for every studio workspace. */

/** Render an icon, explanation, and optional recovery action. */
export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="empty-state">
      <span className="empty-icon">
        <Icon size={24} strokeWidth={1.7} />
      </span>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}
