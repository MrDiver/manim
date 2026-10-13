// The frames the page has fetched, by timeline version, frame number and
// width, and fetching ahead of the playhead.
//
// A frame is asked for by number at a version (/api/frame?f=&v=&s=&w=); the
// server answers which version it rendered (X-Version). Only a frame of
// the version asked for is kept: one rendered after an edit the page has
// not heard of yet is shown, never kept. When the version changes, every
// older frame is dropped. The browser's HTTP cache keeps the same frames
// (the server marks them immutable), so they survive a reload of the page;
// `s` is the server's session, since a new server counts versions anew.
//
// Kept frames are encoded blobs (a JPEG is about 20 times smaller than its
// pixels), the least recently used dropped first once they take more than
// `budget` bytes. Drawing needs decoded pixels: `bitmap` decodes a few
// frames ahead of time (the browser decodes off the main thread), so
// playback draws each frame without waiting.
//
// Fetching ahead (`fetchAhead`), like a video player's buffer: the frames
// from the playhead on, up to `count` frames, nearest first, with a few
// requests open. The page calls it whenever something changed (the
// playhead moved, a frame arrived), playing or not.

const DECODED = 24; // frames kept decoded: the ones about to be drawn

export class FrameCache {
  constructor(budget = 256 * 2 ** 20) {
    this.budget = budget;
    this.version = null;
    this.session = "";
    this.kept = new Map(); // key: { blob, version }, least recently used first
    this.open = new Map(); // key: Promise of the blob
    this.decoded = new Map(); // key: { bitmap (once decoded) }
    this.bytes = 0;
    this.onChange = () => {}; // called when frames are kept or dropped
  }

  static key(version, frame, width) {
    return `${version}/${frame}/${width}`;
  }

  /** Whether the frame is kept. */
  has(version, frame, width) {
    return this.kept.has(FrameCache.key(version, frame, width));
  }

  /** A new server (its versions start over): forget everything. */
  useSession(session) {
    if (session === this.session) return;
    this.session = session;
    this.version = null;
    for (const [key, entry] of this.kept) this.drop(key, entry);
    for (const [key, entry] of this.decoded) this.forget(key, entry);
  }

  /** Forget every frame of another version than `version`. */
  useVersion(version) {
    if (version === this.version) return;
    this.version = version;
    for (const [key, entry] of this.kept) {
      if (entry.version !== version) this.drop(key, entry);
    }
    for (const [key, entry] of this.decoded) {
      if (!key.startsWith(`${version}/`)) this.forget(key, entry);
    }
    this.onChange();
  }

  /** The frame's blob, fetched if it is not kept (shared with a request already open). */
  fetch(version, frame, width) {
    const key = FrameCache.key(version, frame, width);
    const entry = this.kept.get(key);
    if (entry) {
      this.kept.delete(key); // the most recently used is last
      this.kept.set(key, entry);
      return Promise.resolve(entry.blob);
    }
    if (!this.open.has(key)) {
      const request = this.request(key, version, frame, width).finally(() => this.open.delete(key));
      this.open.set(key, request);
    }
    return this.open.get(key);
  }

  async request(key, version, frame, width) {
    const response = await fetch(`/api/frame?f=${frame}&v=${version}&s=${this.session}&w=${width}`);
    if (!response.ok) {
      const { error } = await response.json();
      throw new Error(error);
    }
    const blob = await response.blob();
    if (Number(response.headers.get("X-Version")) === version && version === this.version) {
      this.kept.set(key, { blob, version });
      this.bytes += blob.size;
      this.trim();
      this.onChange();
    }
    return blob;
  }

  /** Keep the `count` frames from `frame` on coming (to `last`, then from 0
   * when looping), nearest first, with at most `requests` open. */
  fetchAhead(version, frame, width, { count, last, loop, requests }) {
    for (let step = 0; step < count && this.open.size < requests; step++) {
      let next = frame + step;
      if (next > last) {
        if (!loop || step > last) break;
        next %= last + 1;
      }
      const key = FrameCache.key(version, next, width);
      if (!this.kept.has(key) && !this.open.has(key)) this.fetch(version, next, width).catch(() => {});
    }
  }

  /** How many frames from `frame` on are kept without a gap (at most `limit`, not past `last`). */
  keptAhead(version, frame, width, last, limit) {
    let count = 0;
    while (count < limit && frame + count <= last && this.has(version, frame + count, width)) count++;
    return count;
  }

  /** The frame's ImageBitmap if it is decoded already, else null (and it
   * is being decoded now, if it is kept). */
  bitmap(version, frame, width) {
    const key = FrameCache.key(version, frame, width);
    const decoded = this.decoded.get(key);
    if (decoded) return decoded.bitmap;
    const entry = this.kept.get(key);
    if (!entry) return null;
    const item = { bitmap: null };
    this.decoded.set(key, item);
    createImageBitmap(entry.blob).then(
      (bitmap) => {
        if (this.decoded.get(key) === item) item.bitmap = bitmap;
        else bitmap.close(); // forgotten meanwhile
      },
      () => this.decoded.delete(key),
    );
    while (this.decoded.size > DECODED) {
      const [oldest, item] = this.decoded.entries().next().value;
      this.forget(oldest, item);
    }
    return null;
  }

  /** The numbers of the kept frames of `version` at `width`, in order. */
  keptFrames(version, width) {
    const numbers = [];
    for (const key of this.kept.keys()) {
      const [keyVersion, frame, keyWidth] = key.split("/").map(Number);
      if (keyVersion === version && keyWidth === width) numbers.push(frame);
    }
    return numbers.sort((a, b) => a - b);
  }

  trim() {
    for (const [key, entry] of this.kept) {
      if (this.bytes <= this.budget) break;
      this.drop(key, entry);
    }
  }

  drop(key, entry) {
    this.kept.delete(key);
    this.bytes -= entry.blob.size;
  }

  forget(key, item) {
    this.decoded.delete(key);
    item.bitmap?.close();
  }
}
