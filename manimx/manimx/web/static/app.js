// The preview page: draws the timeline and shows the frame at the
// playhead. See manimx/web/server.py for the endpoints.
//
// Frames are asked for one at a time. While a request is open, only the
// newest wanted time is remembered; when the frame arrives, that time is
// asked for next. So scrubbing never queues outdated frames on the server.

const elements = {
  title: document.getElementById("title"),
  time: document.getElementById("time"),
  frame: document.getElementById("frame"),
  error: document.getElementById("error"),
  timeline: document.getElementById("timeline"),
  ruler: document.getElementById("ruler"),
  clips: document.getElementById("clips"),
  playhead: document.getElementById("playhead"),
};

const state = {
  timeline: null, // /api/timeline
  time: 0,
  loading: false, // a frame request is open
  wanted: null, // the time to show once it is done
  shownUrl: null,
};

/** Horizontal position of a time, as a percentage of the timeline. */
function percent(time) {
  const duration = state.timeline.duration || 1;
  return (100 * time) / duration;
}

/** The time under a horizontal pixel position on the timeline, snapped to frames. */
function timeAt(clientX) {
  const box = elements.timeline.getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (clientX - box.left) / box.width));
  return snap(fraction * state.timeline.duration);
}

function snap(time) {
  const { fps, duration } = state.timeline;
  return Math.min(duration, Math.max(0, Math.round(time * fps) / fps));
}

function drawTimeline() {
  const { duration, clips, segments } = state.timeline;
  elements.ruler.replaceChildren();
  const step = niceStep(duration);
  for (let t = 0; t <= duration + 1e-9; t += step) {
    const tick = document.createElement("span");
    tick.className = "tick";
    tick.style.left = `${percent(t)}%`;
    tick.textContent = `${Number(t.toFixed(2))} s`;
    elements.ruler.append(tick);
  }

  elements.clips.replaceChildren();
  const order = clips.map((clip, index) => index).sort((a, b) => clips[a].layer - clips[b].layer || a - b);
  const rows = new Map();
  for (const index of order) {
    const clip = clips[index];
    const row = document.createElement("div");
    row.className = "clip";
    const body = document.createElement("div");
    body.className = "clip-body";
    const end = clip.end ?? duration; // a clip that waits for an event runs on
    body.style.left = `${percent(clip.start)}%`;
    body.style.width = `${percent(end - clip.start)}%`;
    body.title = `${clip.name}: ${clip.start.toFixed(2)} – ${clip.end === null ? "waits" : `${end.toFixed(2)} s`}`;
    const name = document.createElement("span");
    name.className = "clip-name";
    name.textContent = clip.name;
    body.append(name);
    row.append(body);
    elements.clips.append(row);
    rows.set(index, row);
  }
  for (const segment of segments) {
    const bar = document.createElement("div");
    bar.className = "segment";
    const end = segment.end ?? duration;
    bar.style.left = `${percent(segment.start)}%`;
    bar.style.width = `${Math.max(0.2, percent(end - segment.start))}%`;
    rows.get(segment.clip)?.append(bar);
  }
}

/** A tick spacing that gives roughly ten labels. */
function niceStep(duration) {
  const raw = Math.max(duration, 1e-3) / 10;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * power).find((step) => step >= raw);
}

function show(time) {
  state.time = time;
  history.replaceState(null, "", `#t=${time.toFixed(3)}`); // a link keeps the place
  elements.playhead.style.left = `${percent(time)}%`;
  elements.time.textContent = `${time.toFixed(2)} s · frame ${Math.round(time * state.timeline.fps)}`;
  if (state.loading) {
    state.wanted = time;
  } else {
    load(time);
  }
}

async function load(time) {
  state.loading = true;
  state.wanted = null;
  const width = Math.min(state.timeline.width, Math.round(elements.frame.clientWidth * devicePixelRatio) || state.timeline.width);
  try {
    const response = await fetch(`/api/frame?t=${time}&w=${width}`);
    if (!response.ok) {
      const { error } = await response.json();
      throw new Error(error);
    }
    const url = URL.createObjectURL(await response.blob());
    await new Promise((resolve) => {
      elements.frame.onload = resolve;
      elements.frame.onerror = resolve;
      elements.frame.src = url;
    });
    if (state.shownUrl) URL.revokeObjectURL(state.shownUrl);
    state.shownUrl = url;
    elements.error.hidden = true;
  } catch (error) {
    elements.error.textContent = `${time.toFixed(2)} s: ${error.message}`;
    elements.error.hidden = false;
  } finally {
    state.loading = false;
    if (state.wanted !== null) load(state.wanted);
  }
}

function listen() {
  const timeline = elements.timeline;
  timeline.addEventListener("pointerdown", (event) => {
    timeline.setPointerCapture(event.pointerId);
    timeline.focus();
    show(timeAt(event.clientX));
  });
  timeline.addEventListener("pointermove", (event) => {
    if (timeline.hasPointerCapture(event.pointerId)) show(timeAt(event.clientX));
  });
  document.addEventListener("keydown", (event) => {
    const { fps, duration } = state.timeline;
    const steps = {
      ArrowLeft: -(event.shiftKey ? 1 : 1 / fps),
      ArrowRight: event.shiftKey ? 1 : 1 / fps,
    };
    let time = null;
    if (event.key in steps) time = snap(state.time + steps[event.key]);
    else if (event.key === "Home") time = 0;
    else if (event.key === "End") time = duration;
    if (time !== null) {
      event.preventDefault();
      show(time);
    }
  });
}

async function start() {
  const response = await fetch("/api/timeline");
  state.timeline = await response.json();
  const { title, width, height } = state.timeline;
  document.title = `${title} · manimx preview`;
  elements.title.textContent = title;
  elements.frame.width = width;
  elements.frame.height = height;
  drawTimeline();
  listen();
  const linked = Number(new URLSearchParams(location.hash.slice(1)).get("t"));
  show(snap(Number.isFinite(linked) ? linked : 0));
}

start();
