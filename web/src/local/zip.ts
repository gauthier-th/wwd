// A minimal ZIP writer (stored, no compression): the bundle is a handful of files, and ONNX
// weights barely compress. Readable by every unzip tool and by Python's zipfile.

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** DOS date and time fields for `date` (local time, as zip tools show it). */
function dosTime(date: Date): [number, number] {
  const time = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
  const day = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  return [time, day];
}

export function zip(files: [string, Uint8Array][], date = new Date()): Uint8Array {
  const names = files.map(([name]) => new TextEncoder().encode(name));
  const [time, day] = dosTime(date);
  const size = files.reduce((n, [, data], i) => n + 30 + names[i].length + data.length + 46 + names[i].length, 22);
  const out = new Uint8Array(size);
  const view = new DataView(out.buffer);
  let pos = 0;
  const offsets: number[] = [];
  const crcs = files.map(([, data]) => crc32(data));
  const header = (sig: number, i: number, central: boolean) => {
    const data = files[i][1];
    view.setUint32(pos, sig, true);
    pos += 4;
    if (central) {
      view.setUint16(pos, 20, true); // version made by
      pos += 2;
    }
    view.setUint16(pos, 20, true); // version needed
    view.setUint16(pos + 2, 0x0800, true); // UTF-8 names
    view.setUint16(pos + 4, 0, true); // stored
    view.setUint16(pos + 6, time, true);
    view.setUint16(pos + 8, day, true);
    view.setUint32(pos + 10, crcs[i], true);
    view.setUint32(pos + 14, data.length, true);
    view.setUint32(pos + 18, data.length, true);
    view.setUint16(pos + 22, names[i].length, true);
    view.setUint16(pos + 24, 0, true); // extra
    pos += 26;
    if (central) {
      view.setUint16(pos, 0, true); // comment
      view.setUint16(pos + 2, 0, true); // disk
      view.setUint16(pos + 4, 0, true); // internal attributes
      view.setUint32(pos + 6, 0, true); // external attributes
      view.setUint32(pos + 10, offsets[i], true);
      pos += 14;
    }
    out.set(names[i], pos);
    pos += names[i].length;
  };
  files.forEach(([, data], i) => {
    offsets.push(pos);
    header(0x04034b50, i, false);
    out.set(data, pos);
    pos += data.length;
  });
  const central = pos;
  files.forEach((_, i) => header(0x02014b50, i, true));
  view.setUint32(pos, 0x06054b50, true);
  view.setUint16(pos + 4, 0, true);
  view.setUint16(pos + 6, 0, true);
  view.setUint16(pos + 8, files.length, true);
  view.setUint16(pos + 10, files.length, true);
  view.setUint32(pos + 12, pos - central, true);
  view.setUint32(pos + 16, central, true);
  view.setUint16(pos + 20, 0, true);
  return out;
}
