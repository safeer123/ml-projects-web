
import { Link } from "react-router-dom";
import projects from "../router/projects";

export default function ({
}) {
    const publishedProjs = projects.filter(p => p.publish);
    return (
        <div className="ml-common-footer-root">
            <hr />
            <p>
                <b>◉ Safeer Chonengal (
                    <a href={"mailto:safeer2c@gmail.com"}>safeer2c@gmail.com</a>
                    )
                </b>
            </p>
            <div>
                {publishedProjs.map((p, i) => (
                    <span key={p.route}>
                        <Link to={p.route}>
                            {p.title}
                        </Link>
                        {i !== publishedProjs.length-1 && " | "}
                    </span>
                ))}
            </div>
        </div>
    )
}