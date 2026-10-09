// Test helpers: the fixtures scripts/make_web_fixtures.py writes under web/.e2e/.

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Fetcher, Manifest } from "./assets";

export const E2E = fileURLToPath(new URL("../../.e2e/", import.meta.url));
export const HAVE_FIXTURES = existsSync(join(E2E, "golden.json"));
export const FIXTURES_HINT = "run `uv run --no-sync python -m scripts.make_web_fixtures` first";

export const readFixture = (...path: string[]) => new Uint8Array(readFileSync(join(E2E, ...path)));
export const golden = () => JSON.parse(readFileSync(join(E2E, "golden.json"), "utf8"));

/** Files of an exported base model version on disk. */
export function diskFetcher(dir: string): Fetcher {
  return async (name) => new Uint8Array(readFileSync(join(dir, name)));
}

export function diskManifest(dir: string): Manifest {
  return JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
}
