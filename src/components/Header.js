
import { Link } from "react-router-dom";

export default function ({
    links = []
}) {
    return (
        <div className="ml-common-header-root">
            {links.map((p, i) => (
                <span key={p.route}>
                    <Link to={p.route}>
                        {p.title}
                    </Link>
                    {i !== links.length - 1 && " | "}
                </span>
            ))}
        </div>
    )
}