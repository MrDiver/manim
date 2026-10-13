// The preview page: the frame at the playhead, playback, and the timeline.
// See manimx/web/server.py for the endpoints and editing.js for editing.
//
// Frames are asked for one at a time. While a request is open, only the
// newest wanted time is remembered; when the frame arrives, that time is
// asked for next. So scrubbing and playback never queue outdated frames on
// the server; playback shows as many frames as the server renders.
//
// Playback runs on the audio's clock when the timeline has sounds
// (/api/audio, mixed once per version), else on the page's clock; Loop
// starts over at the end.
//
// Tracks: one per layer, the highest on top, each clip of that layer on it.
// Under a track, its events: a bar from where a clip starts waiting to the
// event, and the event's pill. A clip's ▸ opens its scene's nodes; under a
// node, the signals animations change; under a signal, those animations.
// Which clips and nodes are open is kept in localStorage by key (clip,
// node path), so a reload keeps them.
//
// Zoom: the lanes are `zoom` times as wide as the space beside the labels
// (the CSS variable --lanes) and the timeline scrolls sideways under its
// sticky labels. Ctrl+wheel zooms around the pointer, the keys and buttons
// around the playhead; a moved playhead is scrolled into view.
//
// Reload on save: /api/changes answers when the timeline changed (an edit,
// or a scene file saved); the page then fetches the timeline again, except
// during a drag, and keeps its time.

import { clipBar, eventPill, initEditing, timelineLoaded, trackLane } from "./editing.js";

const elements = Object.fromEntries(
  [
    "problem", "title", "time", "frame", "error", "timeline", "ruler", "rows", "playhead", "collapse", "indices",
    "labels", "playhead-frame", "play", "mute", "loop", "to-start", "to-end", "previous-frame", "next-frame",
    "zoom-in", "zoom-out", "zoom-fit",
  ].map((id) => [id.replace(/-(\w)/g, (_, letter) => letter.toUpperCase()), document.getElementById(id)]),
);

/** Smallest width of one frame when zoomed in all the way, in pixels. */
const MAX_FRAME_WIDTH = 32;

const state = {
  timeline: null, // /api/timeline
  time: 0, // shown, within the timeline
  intended: 0, // asked for by the user; shown again after a rebuild that makes it fit
  loading: false, // a frame request is open
  wanted: null, // the time to show once it is done
  shownUrl: null,
  open: loadSet("manimx.open"), // keys of the clips and nodes whose tracks are shown
  indices: loadFlag("manimx.indices"), // index labels over the frame
  zoom: 1, // lanes width / the width beside the labels
  dragging: false, // an edit is being dragged: the timeline is not redrawn meanwhile
  scrubbing: false,
  playing: false,
  loop: true,
  muted: loadFlag("manimx.muted"),
  audio: null, // { version, element } of the timeline's sounds
};

// -- time and position ------------------------------------------------------------

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

const frameOf = (time) => Math.round(time * state.timeline.fps);

// -- zoom -------------------------------------------------------------------------------

/** Pixels beside the labels: the lanes' width at zoom 1. */
function fitWidth() {
  const label = parseFloat(getComputedStyle(elements.timeline).getPropertyValue("--label")) || 170;
  return Math.max(100, elements.timeline.clientWidth - label - 12);
}

function maxZoom() {
  const { duration, fps } = state.timeline;
  return Math.max(1, (duration * fps * MAX_FRAME_WIDTH) / fitWidth());
}

/** Zoom to `zoom`, keeping the time under the horizontal position `clientX` there. */
function zoomTo(zoom, clientX) {
  const box = elements.ruler.getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (clientX - box.left) / box.width));
  state.zoom = Math.min(maxZoom(), Math.max(1, zoom));
  layoutLanes();
  drawRuler();
  const label = elements.ruler.offsetLeft;
  const offset = clientX - elements.timeline.getBoundingClientRect().left;
  elements.timeline.scrollLeft = label + fraction * elements.ruler.offsetWidth - offset;
}

/** Zoom by `factor` around the playhead. */
function zoomBy(factor) {
  zoomTo(state.zoom * factor, elements.playhead.getBoundingClientRect().left);
}

function layoutLanes() {
  elements.timeline.style.setProperty("--lanes", `${fitWidth() * state.zoom}px`);
}

/** Scroll the playhead into view if it left the lanes' visible part. */
function revealPlayhead() {
  const timeline = elements.timeline;
  const label = elements.ruler.offsetLeft;
  const x = label + (percent(state.time) / 100) * elements.ruler.offsetWidth;
  const left = timeline.scrollLeft + label;
  const right = timeline.scrollLeft + timeline.clientWidth - 12;
  if (x < left || x > right) timeline.scrollLeft = x - label - (timeline.clientWidth - label) / 2;
}

// -- drawing the timeline ------------------------------------------------------------------

/** Ticks in frames, about one per 80 pixels. */
function drawRuler() {
  const { duration, fps } = state.timeline;
  elements.ruler.replaceChildren();
  const frames = Math.max(1, Math.round(duration * fps));
  const step = niceStep((frames * 80) / (fitWidth() * state.zoom));
  for (let frame = 0; frame <= frames; frame += step) {
    const tick = document.createElement("span");
    tick.className = "tick";
    tick.style.left = `${percent(frame / fps)}%`;
    tick.textContent = frame;
    elements.ruler.append(tick);
  }
}

/** The smallest of 1, 2, 5, 10, 20, 50, ... frames that is at least `raw`. */
function niceStep(raw) {
  const power = 10 ** Math.floor(Math.log10(Math.max(raw, 1)));
  return [1, 2, 5, 10].map((m) => m * power).find((step) => step >= raw);
}

/** The layers that have tracks: every clip's, highest first, and one free one above. */
function layers() {
  const used = [...new Set(state.timeline.clips.map((clip) => clip.layer))].sort((a, b) => b - a);
  return { used, free: used.length ? used[0] + 1 : 0 };
}

function drawTimeline() {
  const { clips, segments, editable } = state.timeline;
  layoutLanes();
  drawRuler();
  elements.rows.replaceChildren();
  const { used, free } = layers();
  if (editable) {
    const { row, lane } = addRow(`track:${free}`, 0, "New track", "track new-track", false);
    row.title = "Drop a scene here to put it on a new track";
    trackLane(lane, free);
  }
  for (const layer of used) {
    const { lane } = addRow(`track:${layer}`, 0, `Track ${layer}`, "track", false);
    trackLane(lane, layer);
    const onTrack = clips.map((clip, index) => ({ clip, index })).filter(({ clip }) => clip.layer === layer);
    for (const { clip, index } of onTrack) {
      lane.append(clipBody(clip, index));
      for (const segment of segments.filter((segment) => segment.clip === index)) {
        const bar = document.createElement("div");
        bar.className = "segment";
        bar.dataset.clip = index;
        bar.style.left = `${percent(segment.start)}%`;
        bar.style.width = `${Math.max(0.2, percent((segment.end ?? state.timeline.duration) - segment.start))}%`;
        lane.append(bar);
      }
    }
    drawEvents(onTrack);
    for (const { clip, index } of onTrack) {
      const key = clipKey(clip, index);
      if (state.open.has(key)) for (const node of clip.nodes) addNode(node, key, 1);
    }
  }
}

function clipKey(clip, index) {
  return `clip:${index}:${clip.name}`;
}

function clipBody(clip, index) {
  const { duration } = state.timeline;
  const body = document.createElement("div");
  body.className = "clip-body";
  const end = clip.end ?? duration; // a clip that waits for an event runs on
  body.style.left = `${percent(clip.start)}%`;
  body.style.width = `${percent(end - clip.start)}%`;
  body.title = `${clip.name}: frames ${frameOf(clip.start)} – ${clip.end === null ? "waits" : frameOf(end)}`;
  if (clip.nodes.length) {
    const key = clipKey(clip, index);
    const expand = document.createElement("button");
    expand.type = "button";
    expand.className = "expand";
    expand.textContent = state.open.has(key) ? "▾" : "▸";
    expand.title = "Show the scene's nodes and animations";
    expand.addEventListener("pointerdown", (event) => event.stopPropagation());
    expand.addEventListener("click", (event) => {
      event.stopPropagation();
      toggle(key);
    });
    body.append(expand);
  }
  body.append(clip.name);
  clipBar(body, clip, index);
  return body;
}

/** The events row under a track: per wait, a bar and a pill (none: no row). */
function drawEvents(onTrack) {
  const waits = onTrack.flatMap(({ clip, index }) => clip.waits.map((wait) => ({ ...wait, clip, index })));
  if (!waits.length) return;
  const { lane } = addRow(`events:${onTrack[0].clip.layer}`, 1, "events", "events", false);
  const fixed = new Set(state.timeline.events.filter((event) => event.fixed).map((event) => event.name));
  for (const wait of waits) {
    const from = wait.from ?? wait.clip.start;
    const never = wait.time === null;
    const bar = document.createElement("div");
    bar.className = `wait${never ? " never" : ""}`;
    bar.style.left = `${percent(from)}%`;
    bar.style.width = `${percent(never ? Math.max(0, state.timeline.duration - from) : Math.max(0, wait.time - from))}%`;
    lane.append(bar);
    const pill = document.createElement("div");
    pill.className = `pill${never ? " never" : fixed.has(wait.event) ? "" : " emitted"}`;
    pill.style.left = `${percent(never ? from : wait.time)}%`;
    pill.textContent = wait.event;
    pill.title = never
      ? `${wait.clip.name} waits for ${wait.event}, which never fires: drag to time it`
      : `${wait.event} at frame ${frameOf(wait.time)}${fixed.has(wait.event) ? "" : " (emitted by a scene)"}: drag to time it`;
    lane.append(pill);
    eventPill(pill, wait, bar);
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
    bar.title = `${animation.name}: frame ${frameOf(animation.start)}${instant ? "" : ` – ${frameOf(animation.end)}`}`;
    if (named && !instant) bar.textContent = animation.name;
    lane.append(bar);
  }
}

/** A row: a label (a toggle when it has rows below) and a lane. */
function addRow(key, depth, name, kind, expandable, type = null) {
  const row = document.createElement("div");
  row.className = `row ${kind}`;
  row.setAttribute("role", "treeitem");
  row.setAttribute("aria-level", depth + 1);
  const label = document.createElement("div");
  label.className = "label";
  label.style.paddingLeft = `${8 + 14 * depth}px`;
  const toggleElement = document.createElement(expandable ? "button" : "span");
  toggleElement.className = "toggle";
  if (expandable) {
    const open = state.open.has(key);
    toggleElement.type = "button";
    toggleElement.setAttribute("aria-expanded", open);
    row.setAttribute("aria-expanded", open);
    toggleElement.addEventListener("click", () => toggle(key));
  }
  const text = document.createElement("span");
  text.className = "name";
  text.textContent = name;
  text.title = type ? `${name} (${type})` : name;
  toggleElement.append(text);
  label.append(toggleElement);
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

function toggle(key) {
  if (state.open.has(key)) state.open.delete(key);
  else state.open.add(key);
  saveSet("manimx.open", state.open);
  drawTimeline();
}

// -- remembered preferences ---------------------------------------------------------------

function loadSet(name) {
  try {
    return new Set(JSON.parse(localStorage.getItem(name) ?? "[]"));
  } catch {
    return new Set();
  }
}

function saveSet(name, set) {
  try {
    localStorage.setItem(name, JSON.stringify([...set]));
  } catch {
    // storage is unavailable: it lasts for this page only
  }
}

function loadFlag(name) {
  try {
    return localStorage.getItem(name) === "on";
  } catch {
    return false;
  }
}

function saveFlag(name, on) {
  try {
    localStorage.setItem(name, on ? "on" : "off");
  } catch {
    // storage is unavailable: it lasts until a reload
  }
}

// -- the frame ---------------------------------------------------------------------------

function show(time, intended = time, reveal = false) {
  state.time = time;
  state.intended = intended;
  history.replaceState(null, "", `#t=${intended.toFixed(3)}`); // a link keeps the place
  elements.playhead.style.left = `${percent(time)}%`;
  elements.playheadFrame.textContent = frameOf(time);
  elements.time.textContent = `${time.toFixed(2)} s · frame ${frameOf(time)}`;
  if (reveal) revealPlayhead();
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
    if (!state.playing) await drawLabels(time);
  } catch (error) {
    elements.error.textContent = `${time.toFixed(2)} s: ${error.message}`;
    elements.error.hidden = false;
  } finally {
    state.loading = false;
    if (state.wanted !== null) load(state.wanted);
  }
}

// -- index labels -------------------------------------------------------------------------

/** Relative luminance of a #rrggbb color, 0 (black) to 1 (white). */
function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** The index of every path over the frame, in its path's color, or nothing with the toggle off. */
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
    tag.style.background = label.color;
    tag.style.color = luminance(label.color) > 0.5 ? "#000" : "#fff";
    elements.labels.append(tag);
  }
}

// -- playback ------------------------------------------------------------------------------

/** The timeline's sounds as an audio element (fetched once per version), or null. */
async function audioElement() {
  const version = state.timeline.version;
  if (state.audio?.version === version) return state.audio.element;
  let element = null;
  try {
    const response = await fetch("/api/audio");
    if (response.status === 200) {
      element = new Audio(URL.createObjectURL(await response.blob()));
      element.preload = "auto";
    }
  } catch {
    element = null; // play without sound
  }
  if (state.audio?.element) state.audio.element.pause();
  state.audio = { version, element };
  return element;
}

async function play() {
  if (state.playing) return;
  if (state.time >= state.timeline.duration - 1e-9) show(0);
  state.playing = true;
  document.body.classList.add("playing");
  const audio = state.muted ? null : await audioElement();
  let origin = performance.now() - state.time * 1000;
  if (audio) {
    audio.currentTime = state.time;
    audio.play().catch(() => {});
  }
  const step = () => {
    if (!state.playing) return;
    let time = audio && !audio.paused ? audio.currentTime : (performance.now() - origin) / 1000;
    if (time >= state.timeline.duration) {
      if (!state.loop) {
        show(state.timeline.duration);
        pause();
        return;
      }
      time = 0;
      origin = performance.now();
      if (audio) audio.currentTime = 0;
    }
    show(time, time, true);
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function pause() {
  state.playing = false;
  document.body.classList.remove("playing");
  state.audio?.element?.pause();
  show(snap(state.time));
}

const togglePlay = () => (state.playing ? pause() : play());

function setMuted(muted) {
  state.muted = muted;
  elements.mute.setAttribute("aria-pressed", String(muted));
  saveFlag("manimx.muted", muted);
  if (muted) state.audio?.element?.pause();
}

/** Move the playhead (pausing playback). */
function seek(time) {
  if (state.playing) pause();
  show(snap(time), snap(time), true);
}

// -- input -------------------------------------------------------------------------------------

function listen() {
  elements.indices.addEventListener("click", () => {
    state.indices = !state.indices;
    elements.indices.setAttribute("aria-pressed", String(state.indices));
    saveFlag("manimx.indices", state.indices);
    drawLabels(state.time);
  });
  const timeline = elements.timeline;
  // Scrubbing: on the ruler, and on node and signal rows.
  timeline.addEventListener("pointerdown", (event) => {
    if (!event.target.closest("#ruler, .row.node .lane, .row.signal .lane")) return;
    timeline.setPointerCapture(event.pointerId);
    timeline.focus();
    state.scrubbing = true;
    seek(timeAt(event.clientX));
  });
  timeline.addEventListener("pointermove", (event) => {
    if (state.scrubbing && timeline.hasPointerCapture(event.pointerId)) show(timeAt(event.clientX));
  });
  timeline.addEventListener("pointerup", () => {
    state.scrubbing = false;
  });
  timeline.addEventListener(
    "wheel",
    (event) => {
      if (!event.ctrlKey && !event.metaKey) return; // plain wheels scroll
      event.preventDefault();
      zoomTo(state.zoom * Math.exp(-event.deltaY * 0.002), event.clientX);
    },
    { passive: false },
  );
  elements.zoomIn.addEventListener("click", () => zoomBy(1.5));
  elements.zoomOut.addEventListener("click", () => zoomBy(1 / 1.5));
  elements.zoomFit.addEventListener("click", () => zoomTo(1, 0));
  elements.play.addEventListener("click", togglePlay);
  elements.mute.addEventListener("click", () => setMuted(!state.muted));
  elements.loop.addEventListener("click", () => {
    state.loop = !state.loop;
    elements.loop.setAttribute("aria-pressed", String(state.loop));
  });
  elements.toStart.addEventListener("click", () => seek(0));
  elements.toEnd.addEventListener("click", () => seek(state.timeline.duration));
  elements.previousFrame.addEventListener("click", () => seek(state.time - 1 / state.timeline.fps));
  elements.nextFrame.addEventListener("click", () => seek(state.time + 1 / state.timeline.fps));
  new ResizeObserver(() => {
    layoutLanes();
    drawRuler();
  }).observe(timeline);
  elements.collapse.addEventListener("click", () => {
    state.open.clear();
    saveSet("manimx.open", state.open);
    drawTimeline();
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input, select, textarea, dialog")) return; // typing, not moving
    const { fps, duration } = state.timeline;
    const keys = {
      " ": togglePlay,
      m: () => setMuted(!state.muted),
      l: () => elements.loop.click(),
      "+": () => zoomBy(1.5),
      "=": () => zoomBy(1.5),
      "-": () => zoomBy(1 / 1.5),
      0: () => zoomTo(1, 0),
      ArrowLeft: () => seek(state.time - (event.shiftKey ? 1 : 1 / fps)),
      ArrowRight: () => seek(state.time + (event.shiftKey ? 1 : 1 / fps)),
      Home: () => seek(0),
      End: () => seek(duration),
    };
    const action = keys[event.key];
    if (action && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      action();
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
  timelineLoaded();
}

/** Wait for the server's changes, forever; show each one at the same time. */
async function followChanges() {
  for (;;) {
    try {
      const response = await fetch(`/api/changes?since=${state.timeline.version}`);
      const { version } = await response.json();
      if (version !== state.timeline.version) {
        while (state.dragging) await new Promise((resolve) => setTimeout(resolve, 50));
        await loadTimeline();
        if (!state.playing) show(snap(state.intended), state.intended);
      }
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 2000)); // the server is gone or restarting
    }
  }
}

async function start() {
  initEditing({
    state,
    elements,
    percent,
    snap,
    timeAt,
    frameOf,
    redraw: drawTimeline,
    refreshFrame: () => show(state.time, state.intended),
    showTime: (time) => show(snap(time)),
  });
  elements.indices.setAttribute("aria-pressed", String(state.indices));
  elements.mute.setAttribute("aria-pressed", String(state.muted));
  await loadTimeline();
  listen();
  const linked = Number(new URLSearchParams(location.hash.slice(1)).get("t"));
  const intended = Number.isFinite(linked) ? linked : 0;
  show(snap(intended), intended);
  followChanges();
}

start();
