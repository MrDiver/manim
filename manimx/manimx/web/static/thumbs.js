// Thumbnails along the clip bars, like a video editor's filmstrip.
//
// A bar gets one tile per THUMB_WIDTH pixels; each shows the clip alone at
// the scene's playing time under the tile's middle (/api/thumbnail, by the
// clip's prepared scene and that time, so moving a clip keeps them, and
// the browser keeps them too). Only tiles scrolled into view are fetched,
// one at a time and never while playing, so they do not hold up frames.

const THUMB_WIDTH = 64; // pixels per tile, at the bar's height a 16:9 frame
const loaded = new Map(); // url: object URL of the image
const queue = [];
let busy = false;
let page = null;

const watcher = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      watcher.unobserve(entry.target);
      request(entry.target);
    }
  },
  { rootMargin: "0px 200px" },
);

export function initThumbs(shared) {
  page = shared;
}

/** Fill `bar` (a clip bar `pixels` wide) with tiles of `clip` (index `index`). */
export function thumbnails(bar, clip, index, pixels) {
  if (!clip.entry || clip.length <= 0) return;
  const { session, fps } = page.state.timeline;
  const strip = document.createElement("div");
  strip.className = "thumbs";
  const count = Math.min(200, Math.max(1, Math.floor(pixels / THUMB_WIDTH)));
  const cutIn = clip.entry.cut?.[0] ?? 0;
  const cutOut = clip.entry.cut?.[1] ?? clip.length;
  const seconds = (clip.end ?? clip.start + (cutOut - cutIn) / clip.entry.speed) - clip.start;
  for (let tile = 0; tile < count; tile++) {
    const offset = ((tile + 0.5) * seconds) / count;
    const at = Math.min(cutOut, cutIn + offset * clip.entry.speed);
    const image = document.createElement("img");
    image.alt = "";
    image.draggable = false;
    image.dataset.url = `/api/thumbnail?clip=${index}&scene=${clip.scene}&at=${Math.round(at * fps) / fps}&s=${session}&w=${THUMB_WIDTH * 2}`;
    const known = loaded.get(image.dataset.url);
    if (known) image.src = known;
    else watcher.observe(image);
    strip.append(image);
  }
  bar.prepend(strip);
}

function request(image) {
  queue.push(image);
  if (!busy) next();
}

async function next() {
  busy = true;
  while (queue.length) {
    while (page.state.playing || page.state.dragging) await new Promise((resolve) => setTimeout(resolve, 250));
    const image = queue.shift();
    if (!image.isConnected) continue; // redrawn meanwhile
    const url = image.dataset.url;
    try {
      if (!loaded.has(url)) {
        const response = await fetch(url);
        if (!response.ok) continue;
        loaded.set(url, URL.createObjectURL(await response.blob()));
      }
      image.src = loaded.get(url);
    } catch {
      // the server is restarting: the next draw asks again
    }
  }
  busy = false;
}
