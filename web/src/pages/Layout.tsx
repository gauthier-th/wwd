import { Link, Outlet, useRouteError } from "react-router";

import { useBaseModels } from "../hooks";

export function Layout() {
  const bases = useBaseModels();
  const base = bases.data?.find((b) => b.default);
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand"><span className="brand-dot" />Wake Word Trainer</Link>
      </header>
      <Outlet />
      <footer>
        {base ? <>Base model {base.version}. </> : null}
        Trained with the Multilingual Spoken Words Corpus (MLCommons, CC-BY 4.0). Your recordings stay on
        this server until you delete them.
      </footer>
    </div>
  );
}

export function ErrorPage() {
  const error = useRouteError();
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand"><span className="brand-dot" />Wake Word Trainer</Link>
      </header>
      <div className="card">
        <h1>Something went wrong</h1>
        <p className="muted">{String((error as Error)?.message ?? error)}</p>
        <Link to="/" className="btn">Back to the start</Link>
      </div>
    </div>
  );
}
