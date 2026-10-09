// Which microphone captures use, and what a page can learn and do about the permission.
// A page can't revoke or reset a permission (Permissions.revoke() never shipped): it can read
// the state, ask again (the browser prompts unless the site is blocked), and point to the
// browser's own site settings.

const DEVICE_KEY = "wwd.micDevice";
export const MIC_CHANGED = "wwd:mic-changed";

export function preferredMic(): string | null {
  try {
    return localStorage.getItem(DEVICE_KEY);
  } catch {
    return null; // storage blocked: the default device
  }
}

export function setPreferredMic(deviceId: string | null): void {
  try {
    if (deviceId) localStorage.setItem(DEVICE_KEY, deviceId);
    else localStorage.removeItem(DEVICE_KEY);
  } catch {
    // storage blocked: the choice lasts until the page closes at most
  }
  window.dispatchEvent(new Event(MIC_CHANGED));
}

const RAW: MediaTrackConstraints = {
  channelCount: 1,
  // Raw signal, like a Pi mic hears it: no browser voice processing.
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
};

/** A capture stream from the chosen microphone, or from the default one if it's gone. */
export async function getMicStream(): Promise<MediaStream> {
  const deviceId = preferredMic();
  if (deviceId) {
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: { ...RAW, deviceId: { exact: deviceId } } });
    } catch (err) {
      const name = (err as Error).name;
      if (name !== "OverconstrainedError" && name !== "NotFoundError") throw err;
      setPreferredMic(null); // unplugged, or the ids changed when the site's data was cleared
    }
  }
  return navigator.mediaDevices.getUserMedia({ audio: RAW });
}

export type MicPermission = PermissionState | "unknown";

/** The permission's live status, or null where the browser doesn't expose it. */
export async function micPermissionStatus(): Promise<PermissionStatus | null> {
  try {
    return await navigator.permissions.query({ name: "microphone" as PermissionName });
  } catch {
    return null;
  }
}

/** Audio inputs. Labels and ids stay empty until the site has been allowed the microphone. */
export async function listMics(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  // Chrome's "default" and "communications" entries alias a real device listed too.
  return devices.filter((d) => d.kind === "audioinput" && d.deviceId !== "default" && d.deviceId !== "communications");
}

/** A user-facing message for a failed capture, or null if it isn't a microphone problem. */
export function micErrorText(err: unknown): string | null {
  switch ((err as Error | undefined)?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Microphone access was refused.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No microphone found.";
    case "NotReadableError":
    case "AbortError":
      return "The microphone couldn't be opened: another app may be using it.";
    default:
      return null;
  }
}

export type Browser = "chrome" | "chrome-android" | "firefox" | "safari" | "ios";

/** Which browser's settings to describe. Every iOS browser is Safari's engine underneath. */
export function browserOf(userAgent: string, touchMac = false): Browser {
  if (/iPhone|iPad|iPod/.test(userAgent) || (touchMac && /Macintosh/.test(userAgent))) return "ios";
  if (/Firefox\//.test(userAgent)) return "firefox";
  if (/Android/.test(userAgent)) return "chrome-android";
  if (/Chrome\/|Chromium\/|Edg\//.test(userAgent)) return "chrome";
  if (/Safari\//.test(userAgent)) return "safari";
  return "chrome";
}
