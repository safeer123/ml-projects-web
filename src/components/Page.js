import Header from "./Header";
import CommonFooter from "./CommonFooter";

export default function({ children, headerLinks, className="" }) {
    return (
        <div className={`ml-project-root-common ${className}`}>
            <Header links={headerLinks} />
            {children}
            <CommonFooter />
        </div>
    )
}