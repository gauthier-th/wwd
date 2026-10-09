// One embedding worker: augments its share of the signals and runs the encoder on them
// (onnxruntime-web, wasm, one thread). The pool (pool.ts) runs several in parallel.

import { Augmenter } from "./augment";
import { loadAugment } from "./assets";
import { EncoderSession, prepare } from "./embed";
import { browserFetcher } from "./fetcher";
import type { FromEmbedder, ToEmbedder } from "./protocol";

let ready: Promise<{ session: EncoderSession; augmenter: Augmenter }> | null = null;

const post = (msg: FromEmbedder, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

self.onmessage = async (e: MessageEvent<ToEmbedder>) => {
  const msg = e.data;
  if (msg.type === "init") {
    const fetcher = browserFetcher(new URL(msg.root), msg.manifest);
    ready = (async () => ({
      session: await EncoderSession.create(await fetcher("encoder.onnx"), msg.manifest.embedding_dim),
      augmenter: new Augmenter(await loadAugment(fetcher, msg.manifest)),
    }))();
    try {
      await ready;
      post({ type: "ready" });
    } catch (err) {
      post({ type: "error", id: null, message: String(err) });
    }
    return;
  }
  try {
    const { session, augmenter } = await ready!;
    const windows = prepare(msg.signals, msg.first, augmenter, msg.augment, msg.seed, msg.hop);
    const embeddings = await session.run(windows);
    post({ type: "result", id: msg.id, embeddings }, [embeddings.buffer]);
  } catch (err) {
    post({ type: "error", id: msg.id, message: String(err) });
  }
};
