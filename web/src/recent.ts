// Wake words created on this device. There are no accounts before M6, so this list is the way
// back to them; it lives in this browser only.

export interface RecentWakeWord {
  id: string;
  name: string;
  language: string;
  created_at: string;
}

const KEY = "wwd.recent";

export function loadRecent(): RecentWakeWord[] {
  try {
    const items = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(items) ? items : [];
  } catch {
    return [];
  }
}

function save(items: RecentWakeWord[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(items.slice(0, 50)));
  } catch {
    // private mode or storage full: the list is a convenience only
  }
}

export function rememberWakeWord(item: RecentWakeWord): void {
  save([item, ...loadRecent().filter((x) => x.id !== item.id)]);
}

export function forgetWakeWord(id: string): void {
  save(loadRecent().filter((x) => x.id !== id));
}
