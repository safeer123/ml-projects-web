import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import projects from "./projects";

export default () => {
  return (
    <Router>
      <div>
        {/* A <Switch> looks through its children <Route>s and
            renders the first one that matches the current URL. */}
        <Switch>
          {projects.map((p) => (
            <Route key={p.route} path={p.route}>
              <p.component />
            </Route>
          ))}
        </Switch>
      </div>
    </Router>
  );
};
