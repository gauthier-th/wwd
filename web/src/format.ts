// Display helpers.

export const pct = (x: number) => `${Math.round(x * 100)}%`;

export function seconds(s: number): string {
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

export function hours(h: number): string {
  if (h >= 1) return `${h.toFixed(1)} h`;
  return `${Math.max(1, Math.round(h * 60))} min`;
}

/** "Deutsch (German)", but just "English". */
export const languageLabel = (l: { name: string; native_name: string }) =>
  l.native_name === l.name ? l.name : `${l.native_name} (${l.name})`;

/** 0.1 -> "1 every 10 h", 2 -> "2 per hour". */
export function falseAlarmRate(perHour: number): string {
  if (perHour === 0) return "none";
  if (perHour < 1) return `1 every ${+(1 / perHour).toFixed(1)} h`;
  return `${+perHour.toFixed(1)} per hour`;
}

export const REASONS: Record<string, string> = {
  too_long: "Too long",
  too_short: "Too short",
  clipped: "Too loud",
  too_quiet: "Too quiet",
};

/** Letters only: "hey jarvis" -> 9. M1 measured short wake words as the weak spot. */
export const letterCount = (name: string) => [...name.normalize("NFC")].filter((c) => /\p{L}/u.test(c)).length;
export const SHORT_WAKE_WORD = 4;

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
