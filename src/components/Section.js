
export default function ({
    title="",
    description="",
    children,
    divider,
    disabled,
}) {
  return (
    <div className={`ml-common-section-root ${disabled ? 'disabled': ''}`}>
        <h3>{title}</h3>
        {description}
        {children}
        {divider && <hr />}
    </div>
  )
}