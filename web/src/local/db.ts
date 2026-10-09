// IndexedDB: where the browser-only app keeps what the server kept in SQLite and on disk.
// One small promise wrapper; every call is its own transaction.

const NAME = "wwd";
const VERSION = 1;

export type Store = "wakewords" | "samples" | "trainings" | "artifacts" | "device_runs" | "field_chunks";

let opened: Promise<IDBDatabase> | null = null;

function db(): Promise<IDBDatabase> {
  return (opened ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(NAME, VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      d.createObjectStore("wakewords", { keyPath: "id" });
      d.createObjectStore("samples", { keyPath: "id" }).createIndex("wakeword_id", "wakeword_id");
      d.createObjectStore("trainings", { keyPath: "id" }).createIndex("wakeword_id", "wakeword_id");
      d.createObjectStore("artifacts"); // key: "<training id>/<name>" or "field/<run id>/<clip>"
      d.createObjectStore("device_runs", { keyPath: "id" }).createIndex("training_id", "training_id");
      d.createObjectStore("field_chunks", { autoIncrement: true }).createIndex("run_id", "run_id");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }));
}

const done = <T>(req: IDBRequest<T>) =>
  new Promise<T>((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

async function store(name: Store, mode: IDBTransactionMode): Promise<IDBObjectStore> {
  return (await db()).transaction(name, mode).objectStore(name);
}

export async function get<T>(name: Store, key: IDBValidKey): Promise<T | undefined> {
  return done((await store(name, "readonly")).get(key)) as Promise<T | undefined>;
}

export async function put(name: Store, value: unknown, key?: IDBValidKey): Promise<void> {
  await done((await store(name, "readwrite")).put(value, key));
}

export async function add(name: Store, value: unknown): Promise<void> {
  await done((await store(name, "readwrite")).add(value));
}

export async function del(name: Store, key: IDBValidKey | IDBKeyRange): Promise<void> {
  await done((await store(name, "readwrite")).delete(key));
}

export async function byIndex<T>(name: Store, index: string, value: IDBValidKey): Promise<T[]> {
  return done((await store(name, "readonly")).index(index).getAll(value)) as Promise<T[]>;
}

export async function all<T>(name: Store): Promise<T[]> {
  return done((await store(name, "readonly")).getAll()) as Promise<T[]>;
}

/** Delete every record of `name` whose `index` is `value`. */
export async function delByIndex(name: Store, index: string, value: IDBValidKey): Promise<void> {
  const s = await store(name, "readwrite");
  const keys = await done(s.index(index).getAllKeys(value));
  await Promise.all(keys.map((k) => done(s.delete(k))));
}

/** Delete every artifact whose key starts with `prefix`. */
export async function delPrefix(prefix: string): Promise<void> {
  await del("artifacts", IDBKeyRange.bound(prefix, `${prefix}￿`));
}
