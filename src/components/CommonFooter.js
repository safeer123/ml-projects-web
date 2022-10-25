
import { Link } from "react-router-dom";
import projects from "../router/projects";

export default function ({
}) {
    return (
        <div className="ml-common-footer-root">
            <hr />
            <p>
                <b>Contact the author: Safeer Chonengal (
                    <a href={"mailto:safeer2c@gmail.com"}>safeer2c@gmail.com</a>
                    )
                </b>
            </p>
            <div>
                {projects.map((p, i) => (
                    <span key={p.route}>
                        <Link to={p.route}>
                            {p.title}
                        </Link>
                        {i !== projects.length-1 && " | "}
                    </span>
                ))}
            </div>
        </div>
    )
}