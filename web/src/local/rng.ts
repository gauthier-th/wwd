// Seeded random numbers for the browser training. They don't reproduce numpy's or torch's
// streams (nothing does across languages): the browser recipe matches the server's in
// distribution, and is reproducible from its seed.

export class Rng {
  private state: number;
  private spare: number | null = null;

  constructor(seed: number) {
    this.state = (seed >>> 0) ^ 0x9e3779b9;
  }

  /** A child generator: deterministic per (this seed, key), independent of call order. */
  static derive(seed: number, key: number): Rng {
    let h = Math.imul((seed >>> 0) ^ 0x85ebca6b, 0xc2b2ae35) ^ Math.imul(key + 1, 0x27d4eb2f);
    h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
    return new Rng(h ^ (h >>> 13));
  }

  /** Uniform in [0, 1) (mulberry32). */
  next(): number {
    let t = (this.state = (this.state + 0x6d2b79f5) | 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.next() * n);
  }

  /** Standard normal (Box-Muller). */
  normal(): number {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    let u = 0;
    while (u === 0) u = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    const a = 2 * Math.PI * this.next();
    this.spare = r * Math.sin(a);
    return r * Math.cos(a);
  }

  /** `k` distinct items of `items`, in random order (partial Fisher-Yates on a copy). */
  sample<T extends ArrayLike<number>>(items: T, k: number): number[] {
    const pool = Array.from(items);
    k = Math.min(k, pool.length);
    for (let i = 0; i < k; i++) {
      const j = i + this.int(pool.length - i);
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, k);
  }
}
