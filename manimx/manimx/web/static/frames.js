// The frames the page has fetched, by timeline version, frame number and
// width, and the requests that fetch frames ahead while playing.
//
// A frame is asked for by number at a version (/api/frame?f=&v=&w=); the
// server answers which version it rendered (X-Version). Only a frame of
// the version asked for is kept: one rendered after an edit the page has
// not heard of yet is shown, never kept. When the version changes, every
// older frame is dropped. The browser's HTTP cache keeps the same frames
// (the server marks them immutable), so they survive a reload of the page.
//
// Kept frames are encoded blobs (a JPEG is about 20 times smaller than its
// pixels), as object URLs an <img> shows at once, the oldest dropped first
// once they take more than `budget` bytes.
//
// Playing: `prefetch` keeps a few requests open for the frames ahead of
// the playhead. It asks for the frame the playhead will be at when the
// answer comes (now plus the measured time a frame takes), so playback
// shows what the server can render on time, and every frame of a second
// loop comes from here.

const PREFETCH_REQUESTS = 3;
const PREFETCH_SECONDS = 2;

export class FrameCache {
  constructor(budget = 256 * 2 ** 20) {
    this.budget = budget;
    this.version = null;
    this.kept = new Map(); // key: { url, size, version }, oldest first
    this.urls = new Set(); // the kept URLs
    this.open = new Map(); // key: Promise of the URL
    this.bytes = 0;
    this.pinned = null; // the URL shown: never revoked while shown
    this.latency = 0.1; // seconds a frame takes, smoothed
  }

  static key(version, frame, width) {
    return `${version}/${frame}/${width}`;
  }

  /** The URL of a kept frame, or null. */
  get(version, frame, width) {
    const key = FrameCache.key(version, frame, width);
    const entry = this.kept.get(key);
    if (!entry) return null;
    this.kept.delete(key); // the most recently used is last
    this.kept.set(key, entry);
    return entry.url;
  }

  /** Whether the frame is kept or on its way. */
  has(version, frame, width) {
    const key = FrameCache.key(version, frame, width);
    return this.kept.has(key) || this.open.has(key);
  }

  /** Forget every frame of another version than `version`. */
  useVersion(version) {
    if (version === this.version) return;
    this.version = version;
    for (const [key, entry] of this.kept) {
      if (entry.version !== version) this.drop(key, entry);
    }
  }

  /** The URL of the frame, fetched if it is not kept (shared with a request already open). */
  fetch(version, frame, width) {
    const key = FrameCache.key(version, frame, width);
    const url = this.get(version, frame, width);
    if (url) return Promise.resolve(url);
    if (!this.open.has(key)) {
      const request = this.request(key, version, frame, width).finally(() => this.open.delete(key));
      this.open.set(key, request);
    }
    return this.open.get(key);
  }

  async request(key, version, frame, width) {
    const started = performance.now();
    const response = await fetch(`/api/frame?f=${frame}&v=${version}&w=${width}`);
    if (!response.ok) {
      const { error } = await response.json();
      throw new Error(error);
    }
    const blob = await response.blob();
    this.latency = 0.8 * this.latency + 0.2 * ((performance.now() - started) / 1000);
    const url = URL.createObjectURL(blob);
    if (Number(response.headers.get("X-Version")) === version && version === this.version) {
      this.kept.set(key, { url, size: blob.size, version });
      this.urls.add(url);
      this.bytes += blob.size;
      this.trim();
    }
    return url;
  }

  /** Keep the frames ahead of `frame` coming while playing. */
  prefetch(version, frame, width, { fps, last, loop }) {
    const lead = Math.ceil(this.latency * fps);
    const ahead = Math.round(PREFETCH_SECONDS * fps);
    for (let step = lead; step <= ahead && this.open.size < PREFETCH_REQUESTS; step++) {
      let next = frame + step;
      if (next > last) {
        if (!loop) break;
        next -= last + 1;
      }
      if (!this.has(version, next, width)) this.fetch(version, next, width).catch(() => {});
    }
  }

  /** The newest kept frame from `frame` back to `oldest`, or null. */
  latest(version, frame, oldest, width) {
    for (let at = frame; at >= oldest; at--) {
      const url = this.get(version, at, width);
      if (url) return { frame: at, url };
    }
    return null;
  }

  /** `url` is shown now (it stays valid while shown); a URL that was not kept is revoked once replaced. */
  show(url) {
    const previous = this.pinned;
    this.pinned = url;
    if (previous && previous !== url && !this.urls.has(previous)) URL.revokeObjectURL(previous);
  }

  trim() {
    for (const [key, entry] of this.kept) {
      if (this.bytes <= this.budget) break;
      this.drop(key, entry);
    }
  }

  drop(key, entry) {
    this.kept.delete(key);
    this.urls.delete(entry.url);
    this.bytes -= entry.size;
    if (entry.url !== this.pinned) URL.revokeObjectURL(entry.url);
  }
}
