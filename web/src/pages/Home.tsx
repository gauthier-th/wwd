import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router";

import { api, ApiError } from "../api";
import { languageLabel, letterCount, SHORT_WAKE_WORD } from "../format";
import { useLanguages } from "../hooks";
import { forgetWakeWord, loadRecent, rememberWakeWord } from "../recent";

export function Home() {
  const languages = useLanguages();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [language, setLanguage] = useState("");
  const [recent, setRecent] = useState(loadRecent);
  const create = useMutation({
    mutationFn: () => api.createWakeWord(name, language),
    onSuccess: (ww) => {
      rememberWakeWord({ id: ww.id, name: ww.name, language: ww.language, created_at: ww.created_at });
      navigate(`/w/${ww.id}`);
    },
  });
  const short = name.trim() !== "" && letterCount(name) <= SHORT_WAKE_WORD;
  const langName = (code: string) => languages.data?.find((l) => l.code === code)?.native_name ?? code;

  return (
    <>
      <section className="card">
        <h1>Make your own wake word</h1>
        <p className="muted">
          Record a few examples of a word or short phrase, and get a detector that runs in the browser, on a
          phone or on a Raspberry Pi. About two minutes of recording, one minute of training.
        </p>
        <form
          className="stack"
          onSubmit={(e) => {
            e.preventDefault();
            create.mutate();
          }}
        >
          <div className="grid-2">
            <div className="field">
              <label htmlFor="ww-name">Wake word</label>
              <input id="ww-name" type="text" value={name} onChange={(e) => setName(e.target.value)}
                     placeholder="e.g. hey jarvis" maxLength={64} autoComplete="off" required />
            </div>
            <div className="field">
              <label htmlFor="ww-lang">Language</label>
              <select id="ww-lang" value={language} onChange={(e) => setLanguage(e.target.value)} required>
                <option value="" disabled>Choose…</option>
                {languages.data?.map((l) => (
                  <option key={l.code} value={l.code}>{languageLabel(l)}</option>
                ))}
              </select>
            </div>
          </div>
          {short && (
            <div className="notice notice-warn">
              Short wake words are harder to detect reliably. A longer word or a two-word phrase works better.
            </div>
          )}
          {create.error && (
            <div className="notice notice-bad" role="alert">
              {create.error instanceof ApiError ? create.error.detail : String(create.error)}
            </div>
          )}
          <div className="row">
            <button className="btn btn-primary" type="submit" disabled={!name.trim() || !language || create.isPending}>
              Start recording
            </button>
          </div>
        </form>
      </section>

      {recent.length > 0 && (
        <section className="card">
          <h2>Your wake words</h2>
          <p className="muted small">Kept in this browser only.</p>
          <ul className="list" data-testid="recent">
            {recent.map((r) => (
              <li key={r.id}>
                <Link to={`/w/${r.id}`}>{r.name}</Link>
                <span className="badge">{langName(r.language)}</span>
                <span className="spacer" />
                <button className="btn btn-small btn-ghost" aria-label={`Remove ${r.name} from this list`}
                        onClick={() => {
                          forgetWakeWord(r.id);
                          setRecent(loadRecent());
                        }}>
                  Hide
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}
