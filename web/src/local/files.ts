// Stored files (paths under /api/) as things the page can use: an object URL for <audio>, or
// a download the browser saves.

import { useEffect, useState } from "react";

import { fetchBytes } from "../api";

/** Save `path`'s bytes as a file named `name`. */
export async function download(path: string, name: string, type = "application/octet-stream"): Promise<void> {
  const url = URL.createObjectURL(new Blob([await fetchBytes(path)], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

/** An object URL for `path` (null until loaded, or when `path` is null); revoked on change. */
export function useObjectUrl(path: string | null, type = "audio/wav"): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!path) return;
    let made: string | null = null;
    let alive = true;
    fetchBytes(path).then((bytes) => {
      if (!alive) return;
      made = URL.createObjectURL(new Blob([bytes], { type }));
      setUrl(made);
    }, () => undefined);
    return () => {
      alive = false;
      if (made) URL.revokeObjectURL(made);
      setUrl(null);
    };
  }, [path, type]);
  return url;
}
