
export default function ({
    title="",
    description="",
    divider
}) {
  return (
    <div className="ml-common-section-root">
        <h3>{title}</h3>
        <p>{description}</p>
        {divider && <hr />}
    </div>
  )
}