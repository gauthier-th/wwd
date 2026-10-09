import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import {
  browserOf,
  getMicStream,
  listMics,
  MIC_CHANGED,
  micErrorText,
  micPermissionStatus,
  preferredMic,
  setPreferredMic,
  type Browser,
  type MicPermission,
} from "../audio/mic";

const OPEN = "wwd:mic-settings";

/** Open the microphone dialog (it lives in the top bar) from anywhere. */
export const openMicSettings = () => window.dispatchEvent(new Event(OPEN));

/** The message for a failed capture, with a way to fix it; null if it isn't a mic problem. */
export function micError(err: unknown): ReactNode | null {
  const text = micErrorText(err);
  return text && <>{text} <MicSettingsLink /></>;
}

export function MicSettingsLink() {
  return (
    <button type="button" className="link-button" onClick={openMicSettings} data-testid="mic-settings-link">
      Microphone settings
    </button>
  );
}

type Shown = MicPermission | "session";

const STATE: Record<Shown, [string, string]> = {
  granted: ["Allowed.", "This site can use your microphone."],
  session: ["Allowed for this visit.", "The browser wasn't asked to remember it, so it will ask again next time."],
  prompt: ["Not decided yet.", "The browser will ask when a recording starts, or now with the button below."],
  denied: ["Blocked.", "The browser won't ask again until you change it in its site settings, below."],
  unknown: ["Not reported by this browser.", "Test the microphone to find out."],
};

const RESET: Record<Browser, string> = {
  chrome: "Click the site-settings icon at the left of the address bar. Set Microphone to Allow, or click " +
    "Reset permission so the browser asks again.",
  "chrome-android": "Tap the icon at the left of the address bar, then Permissions. Set Microphone to Allow, or " +
    "tap Reset permissions so the browser asks again.",
  firefox: "Click the microphone or lock icon in the address bar and remove the microphone permission (×). " +
    "Firefox asks again on the next recording.",
  safari: "In the Safari menu, choose Settings for This Website…, then set Microphone to Ask or Allow.",
  ios: "Tap the page-settings icon (aA) in the address bar, then Website Settings, and set Microphone to Ask " +
    "or Allow.",
};

function useMicPermission(): [MicPermission, () => void] {
  const [state, setState] = useState<MicPermission>("unknown");
  const [generation, setGeneration] = useState(0);
  useEffect(() => {
    let status: PermissionStatus | null = null;
    let alive = true;
    const update = () => {
      if (alive) setState(status?.state ?? "unknown");
    };
    void micPermissionStatus().then((s) => {
      status = s;
      update();
      s?.addEventListener("change", update);
    });
    return () => {
      alive = false;
      status?.removeEventListener("change", update);
    };
  }, [generation]);
  return [state, useCallback(() => setGeneration((g) => g + 1), [])];
}

function useMics(): [MediaDeviceInfo[], () => void] {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const refresh = useCallback(() => {
    listMics().then(setDevices, () => setDevices([]));
  }, []);
  useEffect(() => {
    refresh();
    navigator.mediaDevices?.addEventListener("devicechange", refresh);
    return () => navigator.mediaDevices?.removeEventListener("devicechange", refresh);
  }, [refresh]);
  return [devices, refresh];
}

interface Test {
  label: string;
  stop(): void;
}

/** Top-bar button + dialog: permission state, which microphone, a level test, and how to
 * reset the permission in this browser. */
export function MicSettings() {
  const dialog = useRef<HTMLDialogElement>(null);
  const meter = useRef<HTMLDivElement>(null);
  const [permission, recheck] = useMicPermission();
  const [devices, refreshDevices] = useMics();
  const [chosen, setChosen] = useState(preferredMic);
  const [test, setTest] = useState<Test | null>(null);
  const [error, setError] = useState<string | null>(null);
  const testRef = useRef<Test | null>(null);

  const listed = devices.filter((d) => d.deviceId && d.label);
  // Device labels show access was given. The Permissions API only reports remembered choices
  // (Firefox's one-time grant stays "prompt"), and older Firefox doesn't report at all.
  const state: Shown = listed.length && permission === "prompt" ? "session"
    : listed.length && permission === "unknown" ? "granted" : permission;
  const browser = browserOf(navigator.userAgent, navigator.maxTouchPoints > 1);

  const stopTest = useCallback(() => {
    testRef.current?.stop();
    testRef.current = null;
    setTest(null);
  }, []);

  useEffect(() => {
    const open = () => {
      refreshDevices(); // access may have changed since (a recording, the browser's settings)
      recheck();
      dialog.current?.showModal();
    };
    const changed = () => setChosen(preferredMic());
    window.addEventListener(OPEN, open);
    window.addEventListener(MIC_CHANGED, changed);
    return () => {
      window.removeEventListener(OPEN, open);
      window.removeEventListener(MIC_CHANGED, changed);
      testRef.current?.stop();
    };
  }, [recheck, refreshDevices]);

  async function startTest(context = new AudioContext()) { // created inside the tap (Safari)
    stopTest();
    setError(null);
    try {
      const stream = await getMicStream();
      if (context.state === "suspended") await context.resume();
      const analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      const source = context.createMediaStreamSource(stream);
      source.connect(analyser);
      const buf = new Float32Array(analyser.fftSize);
      let frame = 0;
      const draw = () => {
        analyser.getFloatTimeDomainData(buf);
        let peak = 0;
        for (const v of buf) peak = Math.max(peak, Math.abs(v));
        const db = 20 * Math.log10(peak + 1e-9);
        if (meter.current) {
          meter.current.style.width = `${Math.max(0, Math.min(1, (db + 60) / 60)) * 100}%`;
          meter.current.parentElement?.setAttribute("data-peak", peak.toFixed(4));
        }
        frame = requestAnimationFrame(draw);
      };
      const run: Test = {
        label: stream.getAudioTracks()[0]?.label || "your microphone",
        stop() {
          cancelAnimationFrame(frame);
          source.disconnect();
          stream.getTracks().forEach((t) => t.stop());
          void context.close();
        },
      };
      testRef.current = run;
      setTest(run);
      draw();
    } catch (err) {
      void context.close();
      setError(micErrorText(err) ?? String(err));
    } finally {
      recheck();
      refreshDevices();
    }
  }

  function choose(deviceId: string) {
    setPreferredMic(deviceId || null);
    if (testRef.current) void startTest(); // hear the new choice
  }

  function close() {
    stopTest();
    dialog.current?.close();
  }

  const [title, help] = STATE[state];
  return (
    <>
      <button type="button" className="btn btn-ghost btn-small" onClick={openMicSettings}
              data-testid="mic-button" title={`Microphone: ${title}`}>
        <span className={`light ${state === "granted" || state === "session" ? "on" : state === "denied" ? "bad" : ""}`}
              aria-hidden="true" />
        Microphone
      </button>
      <dialog ref={dialog} className="dialog" onClose={stopTest} aria-labelledby="mic-title" data-testid="mic-dialog">
        <div className="card-head">
          <h2 id="mic-title">Microphone</h2>
          <button type="button" className="btn btn-ghost btn-small" onClick={close} aria-label="Close">✕</button>
        </div>
        <div className="stack">
          <p data-testid="mic-state" data-state={state}><strong>{title}</strong> {help}</p>

          <div className="field">
            <label htmlFor="mic-device">Device</label>
            <select id="mic-device" value={chosen ?? ""} onChange={(e) => choose(e.target.value)}
                    disabled={!listed.length && !chosen}>
              <option value="">System default</option>
              {listed.map((d) => <option key={d.deviceId} value={d.deviceId}>{d.label}</option>)}
              {chosen && !listed.some((d) => d.deviceId === chosen) && (
                <option value={chosen}>Saved microphone (not available now)</option>
              )}
            </select>
            {!listed.length && <span className="muted small">Allow access to choose between your microphones.</span>}
          </div>

          <div className="row">
            <button type="button" className="btn btn-primary" data-testid="mic-test"
                    onClick={() => (test ? stopTest() : void startTest())}>
              {test ? "Stop" : state === "granted" || state === "session" ? "Test the microphone" : "Ask for access"}
            </button>
            {test && <div className="meter" data-testid="mic-meter"><div ref={meter} /></div>}
          </div>
          {test && <p className="muted small" data-testid="mic-hearing">Hearing: {test.label}. Speak to see the level move.</p>}
          {error && <div className="notice notice-bad" role="alert">{error}</div>}

          <details open={state === "denied"} data-testid="mic-reset">
            <summary>Reset or remove the permission</summary>
            <p className="small" style={{ marginTop: 8 }}>
              Pages can't change it themselves. {RESET[browser]} Then reload the page.
            </p>
            <button type="button" className="btn btn-small" onClick={() => location.reload()}>Reload the page</button>
          </details>
        </div>
      </dialog>
    </>
  );
}
