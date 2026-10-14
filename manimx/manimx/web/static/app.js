// The preview page: the frame at the playhead, playback, and the timeline.
// See manimx/web/server.py for the endpoints and editing.js for editing.
//
// Frames are asked for by number at the timeline's version and kept
// (frames.js), so a frame seen once shows at once until an edit. Scrubbing
// asks for one frame at a time: while a request is open, only the newest
// wanted time is remembered and asked for next, so no outdated frames
// queue on the server. Paused or playing, the frames after the playhead
// are fetched ahead (as many as the "ahead" setting says), like a video
// player's buffer; a yellow line over the tracks marks the frames kept.
// The resolution setting renders frames at 1/2, 1/4 or 1/8 of the
// output's width, faster.
//
// Playback draws decoded frames onto a canvas and waits ("buffering")
// when it runs out instead of skipping frames; the timeline's sounds
// (/api/audio, mixed once per version) play through Web Audio, whose
// clock is the playback's. Loop starts over at the end.
//
// Tracks: one per layer, the highest on top, each clip of that layer on it.
// Under a track, its events: a bar from where a clip starts waiting to the
// event, and the event's pill; and its clips' cues, where the event each
// is synced to (a spoken word) puts it. A clip's arrow opens its scene's nodes; under a
// node, the signals animations change; under a signal, those animations.
// Which clips and nodes are open is kept in localStorage by key (clip,
// node path), so a reload keeps them.
//
// Layout: the page fills the window. Splitters resize the scenes sidebar,
// the inspector and the timeline (kept per browser; a double-click resets
// one); the timeline scrolls up and down under its sticky ruler, and the
// frame is as large as fits the space left.
//
// Zoom: at zoom 1 the timeline's duration fills the space beside the
// labels; the lanes (the CSS variable --lanes) are `zoom` times that, and
// the timeline scrolls sideways under its sticky labels. Like in Blender,
// the lanes go on past the end (`state.span` seconds, grown whenever the
// view is scrolled to its right end), shaded there, so clips can be moved
// and dropped later than anything plays now; past the end shows the
// background. Ctrl+wheel zooms around the pointer, the keys and buttons
// around the playhead; a moved playhead is scrolled into view.
//
// Subtitles: the words of the scripts' parts on the timeline, a line at a time,
// over the frame (the Subtitles button or C hides them); a render puts them
// into the video and an .srt file next to it.
//
// Reload on save: /api/changes answers when the timeline changed (an edit,
// or a scene file saved); the page then fetches the timeline again, except
// during a drag, and keeps its time.

import { clipBar, eventPill, initEditing, timelineLoaded, trackLane } from "./editing.js";
import { drawAudioTracks, initAudio } from "./audiotracks.js";
import { FrameCache } from "./frames.js";
import { initRender, renderAvailable } from "./render.js";
import { initThumbs, thumbnails } from "./thumbs.js";

const elements = Object.fromEntries(
  [
    "problem", "title", "time", "frame", "error", "timeline", "ruler", "rows", "playhead", "collapse", "indices",
    "labels", "subtitles", "captions", "playhead-frame", "play", "mute", "loop", "to-start", "to-end", "previous-frame", "next-frame", "after-end", "cached", "stage", "picture", "resolution", "prefetch",
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
  span: 0, // seconds the lanes show: the duration and room after it
  version: 0, // the newest version of the timeline heard of: frames are kept by it
  clock: null, // while playing: the playback clock (see play)
  resolution: loadNumber("manimx.resolution", 2, [1, 2, 4, 8]), // frames are the output's width / this
  prefetch: loadNumber("manimx.prefetch", 240), // frames fetched ahead of the playhead
  open: loadSet("manimx.open"), // keys of the clips and nodes whose tracks are shown
  indices: loadFlag("manimx.indices"), // index labels over the frame
  captions: !loadFlag("manimx.no-captions"), // subtitles over the frame
  zoom: 1, // lanes width / the width beside the labels
  dragging: false, // an edit is being dragged: the timeline is not redrawn meanwhile
  scrubbing: false,
  playing: false,
  loop: true,
  muted: loadFlag("manimx.muted"),
  audio: null, // { version, buffer } of the timeline's sounds, decoded
};

// -- time and position ------------------------------------------------------------

/** Horizontal position of a time, as a percentage of the lanes. */
function percent(time) {
  return (100 * time) / (state.span || 1);
}

/** The span of the lanes for the timeline loaded: at least the duration and some room after it. */
function fitSpan() {
  const { duration } = state.timeline;
  state.span = Math.max(state.span, duration + Math.max(5, 0.25 * duration), ...state.timeline.clips.map((clip) => clip.end ?? clip.start));
}

/** The time under a horizontal pixel position on the lanes, snapped to frames. */
function timeAt(clientX) {
  const box = elements.ruler.getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (clientX - box.left) / box.width));
  return snap(fraction * state.span);
}

/** `time` on a frame, within the lanes (past the timeline's end too). */
function snap(time) {
  const { fps } = state.timeline;
  return Math.min(Math.floor(state.span * fps) / fps, Math.max(0, Math.round(time * fps) / fps));
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
  return Math.max(1, (Math.max(duration, 1) * fps * MAX_FRAME_WIDTH) / fitWidth());
}

/** Zoom to `zoom`, keeping the time under the horizontal position `clientX` there. */
function zoomTo(zoom, clientX) {
  const box = elements.ruler.getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (clientX - box.left) / box.width));
  state.zoom = Math.min(maxZoom(), Math.max(1, zoom));
  layoutLanes();
  drawRuler();
  drawCached();
  const label = elements.ruler.offsetLeft;
  const offset = clientX - elements.timeline.getBoundingClientRect().left;
  elements.timeline.scrollLeft = label + fraction * elements.ruler.offsetWidth - offset;
}

/** Zoom by `factor` around the playhead. */
function zoomBy(factor) {
  zoomTo(state.zoom * factor, elements.playhead.getBoundingClientRect().left);
}

function layoutLanes() {
  const { duration } = state.timeline;
  const secondsWide = duration > 0 ? duration : state.span;
  elements.timeline.style.setProperty("--lanes", `${(fitWidth() * state.zoom * state.span) / secondsWide}px`);
  elements.afterEnd.style.left = `${percent(duration)}%`;
}

/** Scrolled to the lanes' right end: make room for more. */
function growAtEnd() {
  const timeline = elements.timeline;
  if (timeline.scrollLeft + timeline.clientWidth < timeline.scrollWidth - 40) return;
  state.span *= 1.25;
  drawTimeline(); // every bar's place is a share of the span
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
  const { fps } = state.timeline;
  for (const tick of elements.ruler.querySelectorAll(".tick")) tick.remove();
  const frames = Math.max(1, Math.round(state.span * fps));
  const step = niceStep((frames * 80) / elements.ruler.offsetWidth || 1);
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
  // Rebuilding the rows empties them for a moment, which scrolls the timeline back: keep its place.
  const { scrollTop, scrollLeft } = elements.timeline;
  drawRows();
  elements.timeline.scrollTop = scrollTop;
  elements.timeline.scrollLeft = scrollLeft;
}

function drawRows() {
  const { clips, segments, editable } = state.timeline;
  layoutLanes();
  drawRuler();
  drawCached();
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
        bar.style.width = `${Math.max(0.2, percent((segment.end ?? state.span) - segment.start))}%`;
        lane.append(bar);
      }
    }
    drawEvents(onTrack);
    for (const { clip, index } of onTrack) {
      const key = clipKey(clip, index);
      if (state.open.has(key)) for (const node of clip.nodes) addNode(node, key, 1);
    }
  }
  drawAudioTracks();
}

function clipKey(clip, index) {
  return `clip:${index}:${clip.name}`;
}

function clipBody(clip, index) {
  const body = document.createElement("div");
  body.className = "clip-body";
  const end = clip.end ?? state.span; // a clip that waits for an event runs on
  body.style.left = `${percent(clip.start)}%`;
  body.style.width = `${percent(end - clip.start)}%`;
  body.title = `${clip.name}: frames ${frameOf(clip.start)} – ${clip.end === null ? "waits" : frameOf(end)}`;
  if (clip.nodes.length) {
    const key = clipKey(clip, index);
    const expand = document.createElement("button");
    expand.type = "button";
    expand.className = "expand";
    expand.setAttribute("aria-expanded", String(state.open.has(key)));
    expand.title = "Show the scene's nodes and animations";
    expand.addEventListener("pointerdown", (event) => event.stopPropagation());
    expand.addEventListener("click", (event) => {
      event.stopPropagation();
      toggle(key);
    });
    body.append(expand);
  }
  const name = document.createElement("span");
  name.className = "clip-name";
  name.textContent = clip.name;
  body.append(name);
  thumbnails(body, clip, index, (percent(end - clip.start) / 100) * elements.ruler.offsetWidth);
  clipBar(body, clip, index);
  return body;
}

/** The events row under a track: per wait, a bar and a pill; per synced cue, a mark (none: no row). */
function drawEvents(onTrack) {
  const waits = onTrack.flatMap(({ clip, index }) => clip.waits.map((wait) => ({ ...wait, clip, index })));
  const cues = onTrack.flatMap(({ clip }) => (clip.cues ?? []).filter((cue) => cue.time !== null).map((cue) => ({ ...cue, clip })));
  if (!waits.length && !cues.length) return;
  const { lane } = addRow(`events:${onTrack[0].clip.layer}`, 1, "events", "events", false);
  const fixed = new Set(state.timeline.events.filter((event) => event.fixed).map((event) => event.name));
  for (const wait of waits) {
    const from = wait.from ?? wait.clip.start;
    const never = wait.time === null;
    const bar = document.createElement("div");
    bar.className = `wait${never ? " never" : ""}`;
    bar.style.left = `${percent(from)}%`;
    bar.style.width = `${percent(never ? Math.max(0, state.span - from) : Math.max(0, wait.time - from))}%`;
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
  for (const cue of cues) {
    const mark = document.createElement("div");
    mark.className = "cue";
    mark.style.left = `${percent(cue.time)}%`;
    mark.textContent = cue.name;
    mark.title = `${cue.clip.name}'s cue ${cue.name} is at ${cue.event}, frame ${frameOf(cue.time)} (bind it to another event in the clip's panel)`;
    lane.append(mark);
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

// -- the layout: splitters and the frame's size -------------------------------------------

const SIZES = { sidebar: [230, 140], inspector: [300, 200], timeline: [300, 90] }; // default, least (px)

/** The frame as large as fits the stage, at the output's aspect ratio. */
function fitPicture() {
  if (!state.timeline) return;
  const { width, height } = state.timeline;
  const box = elements.stage.getBoundingClientRect();
  const scale = Math.min(box.width / width, box.height / height);
  elements.picture.style.width = `${Math.floor(width * scale)}px`;
  elements.picture.style.height = `${Math.floor(height * scale)}px`;
}

function setSize(name, pixels) {
  const [, least] = SIZES[name];
  const most = name === "timeline" ? innerHeight - 220 : innerWidth / 3;
  const size = Math.round(Math.max(least, Math.min(most, pixels)));
  document.documentElement.style.setProperty(`--${name}`, `${size}px`);
  return size;
}

/** Each splitter resizes the panel named by its data-size; sizes are kept per browser. */
function initSplitters() {
  const saved = loadObject("manimx.layout");
  for (const [name, [size]] of Object.entries(SIZES)) setSize(name, saved[name] ?? size);
  for (const splitter of document.querySelectorAll(".splitter")) {
    const name = splitter.dataset.size;
    const direction = Number(splitter.dataset.direction);
    const vertical = splitter.classList.contains("horizontal");
    const current = () => parseFloat(getComputedStyle(document.documentElement).getPropertyValue(`--${name}`));
    splitter.addEventListener("pointerdown", (down) => {
      splitter.setPointerCapture(down.pointerId);
      splitter.classList.add("dragging");
      const from = current();
      const onMove = (move) => {
        const delta = vertical ? move.clientY - down.clientY : move.clientX - down.clientX;
        saved[name] = setSize(name, from + direction * delta);
      };
      const onUp = () => {
        splitter.classList.remove("dragging");
        splitter.removeEventListener("pointermove", onMove);
        splitter.removeEventListener("pointerup", onUp);
        saveObject("manimx.layout", saved);
      };
      splitter.addEventListener("pointermove", onMove);
      splitter.addEventListener("pointerup", onUp);
    });
    splitter.addEventListener("dblclick", () => {
      delete saved[name];
      setSize(name, SIZES[name][0]);
      saveObject("manimx.layout", saved);
    });
  }
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

function loadObject(name) {
  try {
    const value = JSON.parse(localStorage.getItem(name) ?? "{}");
    return value && typeof value === "object" ? value : {};
  } catch {
    return {};
  }
}

function saveObject(name, value) {
  try {
    localStorage.setItem(name, JSON.stringify(value));
  } catch {
    // storage is unavailable: it lasts until a reload
  }
}

function loadNumber(name, fallback, choices = null) {
  try {
    const value = Number(localStorage.getItem(name) ?? fallback);
    return Number.isFinite(value) && value >= 0 && (!choices || choices.includes(value)) ? value : fallback;
  } catch {
    return fallback;
  }
}

function saveNumber(name, value) {
  try {
    localStorage.setItem(name, String(value));
  } catch {
    // storage is unavailable: it lasts until a reload
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

const frames = new FrameCache();
const REQUESTS = 3; // frame requests open at once while fetching ahead
const BUFFER_SECONDS = 0.5; // kept ahead before playback goes on after running out
const DECODE_AHEAD = 8; // frames decoded before playback draws them

/** Show `time`: the playhead, and (paused) the frame there; while playing, the playback loop draws. */
function show(time, intended = time, reveal = false) {
  state.time = time;
  state.intended = intended;
  if (!state.playing) history.replaceState(null, "", `#t=${intended.toFixed(3)}`); // a link keeps the place
  elements.playhead.style.left = `${percent(time)}%`;
  elements.playheadFrame.textContent = frameOf(time);
  elements.playheadFrame.style.left = `${percent(time)}%`;
  elements.time.textContent = `${time.toFixed(2)} s · frame ${frameOf(time)}`;
  drawSubtitle(time);
  if (reveal) revealPlayhead();
  if (state.playing) return;
  if (state.loading) {
    state.wanted = time;
  } else {
    load(time);
  }
}

/** The subtitle said at `time`, over the frame. */
function drawSubtitle(time) {
  const said = state.captions ? (state.timeline.subtitles ?? []).find((item) => item.start <= time && time < item.end) : null;
  elements.subtitles.hidden = !said;
  if (said && elements.subtitles.textContent !== said.text) elements.subtitles.textContent = said.text;
}

function setCaptions(on) {
  state.captions = on;
  elements.captions.setAttribute("aria-pressed", String(on));
  saveFlag("manimx.no-captions", !on);
  drawSubtitle(state.time);
}

/** The width frames are rendered at: the output's divided by the chosen resolution. */
const frameWidth = () => Math.max(16, Math.round(state.timeline.width / state.resolution));

/** The last frame number of the timeline. */
const lastFrame = () => Math.floor(state.timeline.duration * state.timeline.fps);

/** The server's timeline is at `version` now: frames of older versions are gone. */
function setVersion(version) {
  state.version = Math.max(state.version, version);
  frames.useVersion(state.version);
}

function draw(bitmap) {
  const canvas = elements.frame;
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
}

async function load(time) {
  state.loading = true;
  state.wanted = null;
  try {
    const blob = await frames.fetch(state.version, frameOf(time), frameWidth());
    const bitmap = await createImageBitmap(blob);
    if (!state.playing) draw(bitmap);
    bitmap.close();
    elements.error.hidden = true;
    if (!state.playing) await drawLabels(time);
  } catch (error) {
    elements.error.textContent = `${time.toFixed(2)} s: ${error.message}`;
    elements.error.hidden = false;
  } finally {
    state.loading = false;
    if (state.wanted !== null) load(state.wanted);
    else fetchAhead();
  }
}

// -- fetching ahead ---------------------------------------------------------------------------

/** Keep the frames after the playhead coming (`state.prefetch` of them),
 * playing or not; a frame the playhead waits for while paused goes first. */
function fetchAhead() {
  if (!state.timeline || state.dragging || document.hidden || (!state.playing && state.loading)) return;
  const last = lastFrame();
  const from = Math.min(state.playing && state.clock ? state.clock.frame : frameOf(state.time), last);
  frames.fetchAhead(state.version, from, frameWidth(), { count: state.prefetch, last, loop: state.loop, requests: REQUESTS });
}

/** The yellow line over the tracks: a bar per run of kept frames. */
function drawCached() {
  if (!state.timeline) return;
  const { fps } = state.timeline;
  const runs = [];
  for (const frame of frames.keptFrames(state.version, frameWidth())) {
    const run = runs[runs.length - 1];
    if (run && frame === run[1] + 1) run[1] = frame;
    else runs.push([frame, frame]);
  }
  elements.cached.replaceChildren(
    ...runs.map(([first, last]) => {
      const bar = document.createElement("div");
      bar.style.left = `${percent(first / fps)}%`;
      bar.style.width = `${percent((last + 1 - first) / fps)}%`;
      return bar;
    }),
  );
}

let cachedTimer = null;
frames.onChange = () => {
  fetchAhead(); // a request finished: open the next one
  cachedTimer ??= setTimeout(() => {
    cachedTimer = null;
    drawCached();
  }, 200);
};

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

let audioContext = null; // made on the first play: browsers allow sound after a click or key

/** The timeline's sounds decoded (fetched once per version), or null without sounds. */
async function soundBuffer() {
  const version = state.timeline.version;
  if (state.audio?.version === version) return state.audio.buffer;
  let buffer = null;
  try {
    const response = await fetch("/api/audio");
    if (response.status === 200) buffer = await audioContext.decodeAudioData(await response.arrayBuffer());
  } catch {
    buffer = null; // play without sound
  }
  state.audio = { version, buffer };
  return buffer;
}

/** Play from the playhead, one frame per frame time.
 *
 * With sound, the clock is the sound's: Web Audio starts the mixed sounds
 * at the exact position and its clock (AudioContext.currentTime) runs on
 * from there, so picture and sound cannot drift. Without sound, the clock
 * is the screen refresh's timestamp. Like a video player: when the frame
 * at the playhead is not there yet, the clock (and the sound) stops,
 * "buffering", until BUFFER_SECONDS of frames are kept ahead, instead of
 * skipping frames. The clock starts in the middle of a frame, so frame
 * borders fall between screen refreshes and each frame is drawn for the
 * same number of them.
 */
async function play() {
  if (state.playing) return;
  if (state.time >= state.timeline.duration - 1e-9) show(0);
  state.playing = true;
  document.body.classList.add("playing");
  audioContext ??= new AudioContext();
  audioContext.resume();
  const buffer = await soundBuffer();
  if (!state.playing) return; // paused while the sound loaded
  const clock = { position: (frameOf(state.time) + 0.5) / state.timeline.fps, last: null, frame: frameOf(state.time), shown: null, buffering: false, source: null, anchor: null };
  state.clock = clock;
  /** Start the sound at the clock's position (or stop it); the clock follows it from then. */
  const sound = (on) => {
    clock.source?.stop();
    clock.source = null;
    clock.anchor = null;
    if (!on || !buffer || state.muted || clock.position >= buffer.duration) return;
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    const at = audioContext.currentTime + 0.02; // a moment to schedule it
    source.start(at, clock.position);
    clock.source = source;
    clock.anchor = { at, position: clock.position };
  };
  clock.sound = sound;
  const setBuffering = (buffering) => {
    clock.buffering = buffering;
    elements.stage.classList.toggle("buffering", buffering);
    sound(!buffering);
  };
  setBuffering(false);
  const step = (now) => {
    if (state.clock !== clock) return;
    const { fps, duration } = state.timeline;
    const last = lastFrame();
    const width = frameWidth();
    const version = state.version;
    if (clock.anchor) clock.position = clock.anchor.position + Math.max(0, audioContext.currentTime - clock.anchor.at);
    else if (!clock.buffering && clock.last !== null) clock.position += Math.min(0.1, (now - clock.last) / 1000);
    clock.last = now;
    if (clock.position > duration) {
      if (!state.loop) {
        pause();
        show(duration);
        return;
      }
      clock.position = 0.5 / fps;
      if (!clock.buffering) sound(true);
    }
    clock.frame = Math.min(last, Math.floor(clock.position * fps));
    const kept = frames.keptAhead(version, clock.frame, width, last, Math.round(BUFFER_SECONDS * fps));
    if (clock.buffering && kept >= Math.min(Math.round(BUFFER_SECONDS * fps), last - clock.frame + 1)) setBuffering(false);
    else if (!clock.buffering && kept === 0) setBuffering(true);
    fetchAhead();
    for (let ahead = 1; ahead <= DECODE_AHEAD && clock.frame + ahead <= last; ahead++) frames.bitmap(version, clock.frame + ahead, width);
    const bitmap = frames.bitmap(version, clock.frame, width);
    if (bitmap && clock.shown !== clock.frame) {
      draw(bitmap);
      clock.shown = clock.frame;
      elements.error.hidden = true;
    }
    show(clock.frame / fps, clock.frame / fps, true);
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function pause() {
  state.clock?.sound?.(false);
  state.playing = false;
  state.clock = null;
  document.body.classList.remove("playing");
  elements.stage.classList.remove("buffering");
  show(snap(state.time));
}

const togglePlay = () => (state.playing ? pause() : play());

function setMuted(muted) {
  state.muted = muted;
  elements.mute.setAttribute("aria-pressed", String(muted));
  saveFlag("manimx.muted", muted);
  if (state.clock && !state.clock.buffering) state.clock.sound(!muted);
}

/** The preview's resolution: the output's width divided by `divisor`. */
function setResolution(divisor) {
  state.resolution = divisor;
  elements.resolution.value = String(divisor);
  saveNumber("manimx.resolution", divisor);
  drawCached();
  if (!state.playing) show(state.time, state.intended);
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
  elements.captions.addEventListener("click", () => setCaptions(!state.captions));
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
  initSplitters();
  new ResizeObserver(fitPicture).observe(elements.stage);
  elements.resolution.addEventListener("change", () => setResolution(Number(elements.resolution.value)));
  elements.prefetch.addEventListener("change", () => {
    const count = Math.round(Number(elements.prefetch.value));
    state.prefetch = Number.isFinite(count) && count >= 0 ? count : state.prefetch;
    elements.prefetch.value = state.prefetch;
    saveNumber("manimx.prefetch", state.prefetch);
    fetchAhead();
  });
  document.addEventListener("visibilitychange", fetchAhead);
  elements.toStart.addEventListener("click", () => seek(0));
  elements.toEnd.addEventListener("click", () => seek(state.timeline.duration));
  elements.previousFrame.addEventListener("click", () => seek(state.time - 1 / state.timeline.fps));
  elements.nextFrame.addEventListener("click", () => seek(state.time + 1 / state.timeline.fps));
  new ResizeObserver(() => {
    layoutLanes();
    drawRuler();
    drawCached();
  }).observe(timeline);
  timeline.addEventListener("scroll", growAtEnd, { passive: true });
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
      c: () => setCaptions(!state.captions),
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
  if (state.timeline.session !== frames.session) {
    frames.useSession(state.timeline.session); // the server restarted: its versions start over
    state.version = 0;
  }
  fitSpan();
  setVersion(state.timeline.version);
  const { title, width, height, error } = state.timeline;
  document.title = `${title} · manimx preview`;
  elements.title.textContent = title;
  fitPicture();
  elements.problem.textContent = error ?? "";
  elements.problem.hidden = !error;
  drawTimeline();
  timelineLoaded();
  renderAvailable(state.timeline.editable);
}

/** Wait for the server's changes, forever; show each one at the same time. */
async function followChanges() {
  for (;;) {
    try {
      const response = await fetch(`/api/changes?since=${state.timeline.version}`);
      const { version } = await response.json();
      setVersion(version);
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
  initThumbs({ state });
  initRender({ state });
  initAudio({ state, elements, percent, addRow, play, pause, redraw: drawTimeline, showTime: (time) => show(snap(time)) });
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
    setVersion,
  });
  elements.indices.setAttribute("aria-pressed", String(state.indices));
  elements.captions.setAttribute("aria-pressed", String(state.captions));
  elements.resolution.value = String(state.resolution);
  elements.prefetch.value = state.prefetch;
  elements.mute.setAttribute("aria-pressed", String(state.muted));
  await loadTimeline();
  listen();
  const linked = Number(new URLSearchParams(location.hash.slice(1)).get("t"));
  const intended = Number.isFinite(linked) ? linked : 0;
  show(snap(intended), intended);
  followChanges();
}

start();
