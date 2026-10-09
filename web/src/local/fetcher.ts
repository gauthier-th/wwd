// Base-model files over HTTP, kept in Cache Storage so the ~200 MB download happens once.
// Each URL carries the file's hash, so a re-exported version is fetched again.

import type { Fetcher, Manifest } from "./assets";

const CACHE = "wwd-base";

/** Where the static base-model files live: `base/` next to the app. */
export const baseRoot = () => new URL(`${import.meta.env.BASE_URL}base/`, self.location.origin);

export interface BaseIndex {
  versions: string[];
  default: string;
}

export async function fetchJson<T>(url: URL): Promise<T> {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${url.pathname}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export async function loadManifest(): Promise<{ root: URL; manifest: Manifest }> {
  const index = await fetchJson<BaseIndex>(new URL("index.json", baseRoot()));
  const root = new URL(`${index.default}/`, baseRoot());
  return { root, manifest: await fetchJson<Manifest>(new URL("manifest.json", root)) };
}

async function openCache(): Promise<Cache | null> {
  try {
    return "caches" in self ? await caches.open(CACHE) : null;
  } catch {
    return null; // insecure context, or storage blocked: plain HTTP caching only
  }
}

async function readAll(res: Response, size: number, onBytes?: (n: number) => void): Promise<Uint8Array> {
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const out = new Uint8Array(size);
  const reader = res.body.getReader();
  let received = 0;
  const parts: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length <= size) out.set(value, received);
    else parts.push(value);
    received += value.length;
    onBytes?.(received);
  }
  if (received === size) return out;
  const all = new Uint8Array(received); // the manifest's size was wrong: keep what came
  all.set(out.subarray(0, Math.min(size, received)));
  let off = Math.min(size, received);
  for (const p of parts) {
    all.set(p, off);
    off += p.length;
  }
  return all;
}

export function browserFetcher(root: URL, manifest: Manifest): Fetcher {
  return async (name, onBytes) => {
    const info = manifest.files[name];
    const url = new URL(`${name}?v=${info?.sha256.slice(0, 16) ?? manifest.created_at}`, root).href;
    try {
      const cache = await openCache();
      const cached = await cache?.match(url);
      if (cached) return await readAll(cached, info?.bytes ?? 0, onBytes);
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const bytes = await readAll(res, info?.bytes ?? 0, onBytes);
      // Cached once complete (not through a clone read in parallel, which Chromium can fail).
      await cache?.put(url, new Response(bytes as Uint8Array<ArrayBuffer>, { headers: res.headers }))
        .catch(() => undefined); // over quota: keep going uncached
      return bytes;
    } catch (err) {
      throw new Error(`loading ${name}: ${(err as Error).message}`);
    }
  };
}

/** Bytes of `names` already in Cache Storage (to tell a first download from a reload). */
export async function cachedBytes(root: URL, manifest: Manifest, names: string[]): Promise<number> {
  const cache = await openCache();
  if (!cache) return 0;
  let total = 0;
  for (const name of names) {
    const info = manifest.files[name];
    if (info && (await cache.match(new URL(`${name}?v=${info.sha256.slice(0, 16)}`, root).href))) total += info.bytes;
  }
  return total;
}

/** Drop cached files that the current manifest no longer lists (older exports). */
export async function pruneCache(root: URL, manifest: Manifest): Promise<void> {
  const cache = await openCache();
  if (!cache) return;
  const keep = new Set(Object.entries(manifest.files).map(([n, f]) =>
    new URL(`${n}?v=${f.sha256.slice(0, 16)}`, root).href));
  for (const req of await cache.keys()) if (!keep.has(req.url)) await cache.delete(req);
}
