// The preview page: draws the timeline and shows the frame at the
// playhead. See manimx/web/server.py for the endpoints.
//
// Frames are asked for one at a time. While a request is open, only the
// newest wanted time is remembered; when the frame arrives, that time is
// asked for next. So scrubbing never queues outdated frames on the server.
//
// Tracks: under each clip, its scene's nodes as a tree; under a node, the
// signals animations change; under a signal, those animations. A closed
// node's lane shows every animation in its subtree. Which rows are open is
// kept in localStorage by key (clip, node path, signal name), so a reload
// on save keeps them.
//
// Indices: with the toggle on, the page asks /api/labels for every frame
// it shows and draws each path's index in its group over the image (the
// name and group on hover). The toggle is kept in localStorage.
//
// Reload on save: /api/changes answers when the server rebuilt the
// timeline (or failed to); the page then fetches the timeline again and
// shows the same time, or the build error above the frame.

const elements = {
  problem: document.getElementById("problem"),
  title: document.getElementById("title"),
  time: document.getElementById("time"),
  frame: document.getElementById("frame"),
  error: document.getElementById("error"),
  timeline: document.getElementById("timeline"),
  ruler: document.getElementById("ruler"),
  rows: document.getElementById("rows"),
  playhead: document.getElementById("playhead"),
  collapse: document.getElementById("collapse"),
  indices: document.getElementById("indices"),
  labels: document.getElementById("labels"),
};

const state = {
  timeline: null, // /api/timeline
  time: 0, // shown, within the timeline
  intended: 0, // asked for by the user; shown again after a rebuild that makes it fit
  loading: false, // a frame request is open
  wanted: null, // the time to show once it is done
  shownUrl: null,
  open: loadOpen(), // keys of the rows whose tracks are shown
  indices: loadIndices(), // index labels over the frame
};

/** Horizontal position of a time, as a percentage of the timeline. */
function percent(time) {
  const duration = state.timeline.duration || 1;
  return (100 * time) / duration;
}

/** The time under a horizontal pixel position on the lanes, snapped to frames. */
function timeAt(clientX) {
  const box = elements.ruler.getBoundingClientRect();
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

  elements.rows.replaceChildren();
  const order = clips.map((clip, index) => index).sort((a, b) => clips[a].layer - clips[b].layer || a - b);
  for (const index of order) {
    const clip = clips[index];
    const key = `clip:${index}:${clip.name}`;
    const { row, lane } = addRow(key, 0, clip.name, "clip", clip.nodes.length > 0);
    const body = document.createElement("div");
    body.className = "clip-body";
    const end = clip.end ?? duration; // a clip that waits for an event runs on
    body.style.left = `${percent(clip.start)}%`;
    body.style.width = `${percent(end - clip.start)}%`;
    body.title = `${clip.name}: ${clip.start.toFixed(2)} – ${clip.end === null ? "waits" : `${end.toFixed(2)} s`}`;
    lane.append(body);
    for (const segment of segments.filter((segment) => segment.clip === index)) {
      const bar = document.createElement("div");
      bar.className = "segment";
      bar.style.left = `${percent(segment.start)}%`;
      bar.style.width = `${Math.max(0.2, percent((segment.end ?? duration) - segment.start))}%`;
      lane.append(bar);
    }
    if (state.open.has(key)) {
      for (const node of clip.nodes) addNode(node, key, 1);
    }
    row.dataset.clip = index;
  }
}

/** A node's row, and when open its signals' rows and its children. */
function addNode(node, parentKey, depth) {
  const key = `${parentKey}/${node.name}`;
  const expandable = node.signals.length > 0 || node.children.length > 0;
  const { lane } = addRow(key, depth, node.name, "node", expandable, node.type);
  addBars(lane, unique(subtreeAnimations(node)));
  if (!state.open.has(key)) return;
  for (const signal of node.signals) {
    const signalRow = addRow(`${key}#${signal.name}`, depth + 1, signal.name, "signal", false);
    addBars(signalRow.lane, signal.animations, true);
  }
  for (const child of node.children) addNode(child, key, depth + 1);
}

function subtreeAnimations(node) {
  return [...node.signals.flatMap((signal) => signal.animations), ...node.children.flatMap(subtreeAnimations)];
}

/** One bar per animation (equal name, start and end shown once). */
function unique(animations) {
  const seen = new Map();
  for (const animation of animations) seen.set(`${animation.name}@${animation.start}@${animation.end}`, animation);
  return [...seen.values()];
}

function addBars(lane, animations, named = false) {
  for (const animation of animations) {
    const bar = document.createElement("div");
    const instant = animation.end - animation.start < 1e-9;
    bar.className = instant ? "instant" : "animation";
    bar.style.left = `${percent(animation.start)}%`;
    if (!instant) bar.style.width = `${percent(animation.end - animation.start)}%`;
    bar.title = `${animation.name}: ${animation.start.toFixed(2)}${instant ? "" : ` – ${animation.end.toFixed(2)}`} s`;
    if (named && !instant) bar.textContent = animation.name;
    lane.append(bar);
  }
}

/** A row: a label (a toggle when it has tracks below) and a lane. */
function addRow(key, depth, name, kind, expandable, type = null) {
  const row = document.createElement("div");
  row.className = `row ${kind}`;
  row.setAttribute("role", "treeitem");
  row.setAttribute("aria-level", depth + 1);
  const label = document.createElement("div");
  label.className = "label";
  label.style.paddingLeft = `${8 + 14 * depth}px`;
  const toggle = document.createElement(expandable ? "button" : "span");
  toggle.className = "toggle";
  if (expandable) {
    const open = state.open.has(key);
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", open);
    row.setAttribute("aria-expanded", open);
    toggle.addEventListener("click", () => {
      if (state.open.has(key)) state.open.delete(key);
      else state.open.add(key);
      saveOpen();
      drawTimeline();
    });
  }
  const text = document.createElement("span");
  text.className = "name";
  text.textContent = name;
  text.title = type ? `${name} (${type})` : name;
  toggle.append(text);
  label.append(toggle);
  if (type) {
    const badge = document.createElement("span");
    badge.className = "type";
    badge.textContent = type;
    label.append(badge);
  }
  const lane = document.createElement("div");
  lane.className = "lane";
  row.append(label, lane);
  elements.rows.append(row);
  return { row, lane };
}

/** Open tracks are remembered per page (a reload keeps them). */
function loadOpen() {
  try {
    return new Set(JSON.parse(localStorage.getItem("manimx.open") ?? "[]"));
  } catch {
    return new Set();
  }
}

function saveOpen() {
  try {
    localStorage.setItem("manimx.open", JSON.stringify([...state.open]));
  } catch {
    // storage is unavailable: the tracks stay open for this page only
  }
}

/** A tick spacing that gives roughly ten labels. */
function niceStep(duration) {
  const raw = Math.max(duration, 1e-3) / 10;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * power).find((step) => step >= raw);
}

function show(time, intended = time) {
  state.time = time;
  state.intended = intended;
  history.replaceState(null, "", `#t=${intended.toFixed(3)}`); // a link keeps the place
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
    await drawLabels(time);
  } catch (error) {
    elements.error.textContent = `${time.toFixed(2)} s: ${error.message}`;
    elements.error.hidden = false;
  } finally {
    state.loading = false;
    if (state.wanted !== null) load(state.wanted);
  }
}

function loadIndices() {
  try {
    return localStorage.getItem("manimx.indices") === "on";
  } catch {
    return false;
  }
}

function showIndices(on) {
  state.indices = on;
  elements.indices.setAttribute("aria-pressed", String(on));
  try {
    localStorage.setItem("manimx.indices", on ? "on" : "off");
  } catch {
    // no storage: the toggle only lasts until a reload
  }
}

/** The index of every path over the frame, or nothing with the toggle off. */
async function drawLabels(time) {
  elements.labels.replaceChildren();
  if (!state.indices) return;
  const response = await fetch(`/api/labels?t=${time}`);
  if (!response.ok) return;
  for (const label of await response.json()) {
    const tag = document.createElement("span");
    tag.textContent = label.index;
    tag.title = `${label.name} (${label.index} in ${label.group})`;
    tag.style.left = `${100 * label.x}%`;
    tag.style.top = `${100 * label.y}%`;
    elements.labels.append(tag);
  }
}

function listen() {
  elements.indices.addEventListener("click", () => {
    showIndices(!state.indices);
    drawLabels(state.time);
  });
  const timeline = elements.timeline;
  timeline.addEventListener("pointerdown", (event) => {
    if (!event.target.closest(".lane")) return; // the labels toggle tracks
    timeline.setPointerCapture(event.pointerId);
    timeline.focus();
    show(timeAt(event.clientX));
  });
  timeline.addEventListener("pointermove", (event) => {
    if (timeline.hasPointerCapture(event.pointerId)) show(timeAt(event.clientX));
  });
  elements.collapse.addEventListener("click", () => {
    state.open.clear();
    saveOpen();
    drawTimeline();
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

async function loadTimeline() {
  const response = await fetch("/api/timeline");
  state.timeline = await response.json();
  const { title, width, height, error } = state.timeline;
  document.title = `${title} · manimx preview`;
  elements.title.textContent = title;
  elements.frame.width = width;
  elements.frame.height = height;
  elements.problem.textContent = error ?? "";
  elements.problem.hidden = !error;
  drawTimeline();
}

/** Wait for the server's rebuilds, forever; show each one at the same time. */
async function followChanges() {
  for (;;) {
    try {
      const response = await fetch(`/api/changes?since=${state.timeline.version}`);
      const { version } = await response.json();
      if (version !== state.timeline.version) {
        await loadTimeline();
        show(snap(state.intended), state.intended);
      }
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 2000)); // the server is gone or restarting
    }
  }
}

async function start() {
  showIndices(state.indices);
  await loadTimeline();
  listen();
  const linked = Number(new URLSearchParams(location.hash.slice(1)).get("t"));
  const intended = Number.isFinite(linked) ? linked : 0;
  show(snap(intended), intended);
  followChanges();
}

start();
