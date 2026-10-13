// Editing the project from the preview page (manimx/web/editing.py), live:
// every edit changes the server's timeline at once and the frame at the
// playhead shows it; the server saves manimx.json a moment later.
//
// Clips: drag a clip to move it, up or down onto another track to change
// its layer (all selected clips move together), drag the handle at either
// end to cut its start or end (and back out up to the scene's length).
// Click selects, Shift or Ctrl click adds to the selection, dragging on
// empty track space selects every clip the rectangle touches, Escape
// clears it, Delete removes the selected clips. Drag a scene from the
// sidebar onto a track to add a clip of it there. The clip panel types
// start, layer, cut and speed. Events: drag a pill on a track's events row
// to time the event. The settings dialog changes the project's settings.
// Ctrl+Z undoes any edit (a whole drag at once), Ctrl+Shift+Z or Ctrl+Y
// redoes it; the server keeps the history (manimx/web/commands.py).
// S splits the selected clips (or, with none selected, every clip) under
// the playhead, Ctrl+D puts a copy after each selected clip. Ctrl+C copies
// the selected clips (Ctrl+X also removes them), Ctrl+V pastes them at the
// playhead, on their tracks, as far apart as they were; the clipboard is
// kept in the browser, so it works across reloads and tabs.
//
// Snapping: a dragged clip's start or end, or a dragged cut, snaps to the
// other clips' starts and ends, the playhead, the events and 0 when it
// comes within SNAP_PIXELS of one; a line shows where. Alt drags freely.
//
// While dragging, the page sends the edit at every step (one request at a
// time, then the newest) and asks for the frame at the playhead, or at the
// cut while cutting, or at the event while timing one, so the output shows
// where things land. app.js does not redraw the timeline during a drag
// (state.dragging); it fetches the edited timeline once the drag ends.

let page = null; // what app.js shares: state, elements and its helpers
const selected = new Set(); // indices of the selected clips
const SCENE_TYPE = "application/x-manimx-scene";
const SNAP_PIXELS = 8;

const panels = {
  inspector: document.getElementById("inspector"),
  sidebar: document.getElementById("sidebar"),
  scenes: document.getElementById("scene-list"),
  clip: document.getElementById("clip-editor"),
  events: document.getElementById("events-editor"),
  message: document.getElementById("edit-message"),
  settingsButton: document.getElementById("settings-button"),
  undo: document.getElementById("undo"),
  redo: document.getElementById("redo"),
  settings: document.getElementById("settings"),
  settingsForm: document.getElementById("settings-form"),
  marquee: document.getElementById("marquee"),
};

export function initEditing(shared) {
  page = shared;
  panels.settingsButton.addEventListener("click", openSettings);
  panels.settings.addEventListener("close", () => panels.settingsForm.replaceChildren());
  panels.undo.addEventListener("click", () => post("/api/undo", {}));
  panels.redo.addEventListener("click", () => post("/api/redo", {}));
  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input, select, textarea, dialog") || !page.state.timeline?.editable) return;
    const key = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && (key === "z" || key === "y")) {
      event.preventDefault();
      post(key === "y" || event.shiftKey ? "/api/redo" : "/api/undo", {});
      return;
    }
    if (event.key === "Escape" && selected.size) {
      selected.clear();
      page.redraw();
      drawClipPanel();
    }
    if ((event.ctrlKey || event.metaKey) && (key === "c" || key === "x") && selected.size) {
      event.preventDefault();
      copy(key === "x");
      return;
    }
    if ((event.ctrlKey || event.metaKey) && key === "v") {
      event.preventDefault();
      paste();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && key === "d") {
      event.preventDefault(); // not the browser's bookmark
      duplicate();
      return;
    }
    if (key === "s" && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      split();
      return;
    }
    if ((event.key === "Delete" || event.key === "Backspace") && selected.size) {
      event.preventDefault();
      const indices = [...selected];
      selected.clear();
      post("/api/clip/remove", { indices });
    }
  });
}

/** The clips under the playhead (of the selection, if there is one) split there. */
function split() {
  const time = page.state.time;
  const under = page.state.timeline.clips
    .map((clip, index) => ({ clip, index }))
    .filter(({ clip, index }) => clip.entry && clip.start < time && time < (clip.end ?? Infinity) && (!selected.size || selected.has(index)))
    .map(({ index }) => index);
  if (under.length) post("/api/clip/split", { indices: under, time });
  else say("No clip to split under the playhead.", true);
}

const CLIPBOARD = "manimx.clipboard";

/** The selected clips to the clipboard: scene, cut, speed, track, and time after the first. */
function copy(andRemove) {
  const { clips } = page.state.timeline;
  const chosen = [...selected].map((index) => clips[index]).filter((clip) => clip.entry);
  const first = Math.min(...chosen.map((clip) => clip.start));
  const copied = chosen.map((clip) => ({ scene: clip.entry.scene, offset: clip.start - first, layer: clip.layer, cut: clip.entry.cut, speed: clip.entry.speed }));
  try {
    localStorage.setItem(CLIPBOARD, JSON.stringify(copied));
  } catch {
    return say("The browser keeps no clipboard here (storage is off).", true);
  }
  if (andRemove) {
    const indices = [...selected];
    selected.clear();
    post("/api/clip/remove", { indices });
  } else {
    say(`Copied ${copied.length === 1 ? "a clip" : `${copied.length} clips`}.`);
  }
}

function paste() {
  let copied = [];
  try {
    copied = JSON.parse(localStorage.getItem(CLIPBOARD) ?? "[]");
  } catch {
    copied = [];
  }
  if (!copied.length) return say("Nothing copied yet: select clips and press Ctrl+C.", true);
  const time = page.state.time;
  say(`Pasting ${copied.length === 1 ? "a clip" : `${copied.length} clips`}…`);
  post("/api/clip/add", { clips: copied.map(({ offset, ...clip }) => ({ ...clip, start: Math.round((time + offset) * 1e6) / 1e6 })) });
}

function duplicate() {
  if (selected.size) post("/api/clip/duplicate", { indices: [...selected] });
}

/** Called by app.js whenever the timeline was (re)loaded. */
export function timelineLoaded() {
  const { editable, clips } = page.state.timeline;
  panels.inspector.hidden = !editable;
  panels.sidebar.hidden = !editable;
  panels.settingsButton.hidden = !editable;
  panels.undo.hidden = panels.redo.hidden = !editable;
  if (!editable) return;
  showHistory(page.state.timeline.history);
  for (const index of [...selected]) if (!clips[index]) selected.delete(index);
  drawClipPanel();
  drawEventsPanel();
  drawScenes();
}

// -- sending edits ------------------------------------------------------------

async function post(route, body) {
  try {
    const response = await fetch(route, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.error);
    page.setVersion(answer.version); // the next frames are of the edited timeline
    showHistory(answer.history);
    say("Live; saved to manimx.json a moment after the last edit.");
    return true;
  } catch (error) {
    say(error.message, true);
    return false;
  }
}

/** The undo and redo buttons: what each would do, or disabled. */
function showHistory(history) {
  for (const [button, action, keys] of [
    [panels.undo, history?.undo, "Ctrl+Z"],
    [panels.redo, history?.redo, "Ctrl+Shift+Z"],
  ]) {
    button.disabled = !action;
    button.title = action ? `${button === panels.undo ? "Undo" : "Redo"} ${action} (${keys})` : `Nothing to ${button === panels.undo ? "undo" : "redo"}`;
  }
}

function say(text, problem = false) {
  panels.message.textContent = text;
  panels.message.classList.toggle("problem", problem);
}

const editClips = (clips) => post("/api/clip", { clips });
const editClip = (index, changes) => editClips([{ index, ...changes }]);
const setEvent = (name, time) => post("/api/event", { name, time });

let gestures = 0;

/** Sends live edits while a drag goes on: one request at a time, then the
 * newest. All carry one gesture id, so one undo takes back the whole drag. */
function liveSender(route, onSent) {
  let busy = false;
  let next = null;
  const gesture = `${Date.now()}-${++gestures}`;
  const send = async (body) => {
    if (busy) {
      next = body;
      return;
    }
    busy = true;
    await post(route, { ...body, gesture });
    onSent();
    busy = false;
    if (next !== null) {
      const body = next;
      next = null;
      send(body);
    }
  };
  send.idle = async () => {
    while (busy) await new Promise((resolve) => setTimeout(resolve, 10));
  };
  return send;
}

// -- the scenes sidebar ----------------------------------------------------------

async function drawScenes() {
  const response = await fetch("/api/scenes");
  if (!response.ok) return;
  panels.scenes.replaceChildren();
  for (const scene of await response.json()) {
    const item = document.createElement("li");
    item.draggable = true;
    item.title = `${scene.module}.${scene.class}: drag onto a track`;
    const name = document.createElement("span");
    name.textContent = scene.name;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = scene.clips ? `×${scene.clips}` : "";
    count.title = `${scene.clips} clip${scene.clips === 1 ? "" : "s"} on the timeline`;
    item.append(name, count);
    item.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData(SCENE_TYPE, scene.name);
      event.dataTransfer.effectAllowed = "copy";
    });
    panels.scenes.append(item);
  }
}

// -- tracks: dropping scenes, selecting with a rectangle ------------------------------------

/** A track's lane (app.js calls this for every track, `layer` its layer). */
export function trackLane(lane, layer) {
  if (!page.state.timeline.editable) return;
  lane.dataset.layer = layer;
  const row = lane.parentElement;
  lane.addEventListener("dragover", (event) => {
    if (!event.dataTransfer.types.includes(SCENE_TYPE)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    row.classList.add("drop");
  });
  lane.addEventListener("dragleave", () => row.classList.remove("drop"));
  lane.addEventListener("drop", async (event) => {
    row.classList.remove("drop");
    const scene = event.dataTransfer.getData(SCENE_TYPE);
    if (!scene) return;
    event.preventDefault();
    const start = onFrame(timeAtUnclamped(event.clientX));
    say(`Preparing ${scene}…`);
    if (await post("/api/clip/add", { scene, start, layer })) page.showTime(start);
  });
  lane.addEventListener("pointerdown", (event) => {
    if (event.target !== lane && !event.target.classList.contains("segment")) return;
    event.stopPropagation();
    lane.setPointerCapture(event.pointerId);
    const additive = event.shiftKey || event.ctrlKey || event.metaKey;
    const before = new Set(additive ? selected : []);
    const origin = { x: event.clientX, y: event.clientY };
    let dragged = false;
    const onMove = (move) => {
      dragged ||= Math.abs(move.clientX - origin.x) + Math.abs(move.clientY - origin.y) > 3;
      const box = rectangle(origin, { x: move.clientX, y: move.clientY });
      showMarquee(box);
      selected.clear();
      for (const index of before) selected.add(index);
      for (const bar of document.querySelectorAll(".clip-body.editable")) {
        const bounds = bar.getBoundingClientRect();
        if (bounds.right >= box.left && bounds.left <= box.right && bounds.bottom >= box.top && bounds.top <= box.bottom) selected.add(Number(bar.dataset.index));
      }
      markSelected();
    };
    const onUp = (up) => {
      lane.removeEventListener("pointermove", onMove);
      lane.removeEventListener("pointerup", onUp);
      panels.marquee.hidden = true;
      if (!dragged) {
        if (!additive) selected.clear(); // a click on empty space: and it moves the playhead there
        page.showTime(timeAtUnclamped(up.clientX));
      }
      page.redraw();
      drawClipPanel();
    };
    lane.addEventListener("pointermove", onMove);
    lane.addEventListener("pointerup", onUp);
  });
}

/** The time under `clientX` on the lanes, also past the timeline's end. */
function timeAtUnclamped(clientX) {
  const box = page.elements.ruler.getBoundingClientRect();
  return Math.max(0, ((clientX - box.left) / box.width) * page.state.span);
}

function rectangle(a, b) {
  return { left: Math.min(a.x, b.x), right: Math.max(a.x, b.x), top: Math.min(a.y, b.y), bottom: Math.max(a.y, b.y) };
}

function showMarquee(box) {
  const marquee = panels.marquee;
  marquee.hidden = false;
  marquee.style.left = `${box.left + scrollX}px`;
  marquee.style.top = `${box.top + scrollY}px`;
  marquee.style.width = `${box.right - box.left}px`;
  marquee.style.height = `${box.bottom - box.top}px`;
}

function markSelected() {
  for (const bar of document.querySelectorAll(".clip-body.editable")) bar.classList.toggle("selected", selected.has(Number(bar.dataset.index)));
}

// -- clip bars ------------------------------------------------------------------

/** Make a clip's bar editable (app.js calls this for every clip bar). */
export function clipBar(body, clip, index) {
  if (!page.state.timeline.editable || !clip.entry) return;
  body.classList.add("editable");
  body.classList.toggle("selected", selected.has(index));
  body.dataset.index = index;
  const handles = clip.end === null ? ["in"] : ["in", "out"]; // a waiting clip has no end to cut
  for (const edge of handles) {
    const handle = document.createElement("div");
    handle.className = `handle ${edge}`;
    handle.title = edge === "in" ? "Drag to cut the start" : "Drag to cut the end";
    handle.addEventListener("pointerdown", (event) => startDrag(event, edge, index));
    body.append(handle);
  }
  body.addEventListener("pointerdown", (event) => startDrag(event, "move", index));
}

/** The layer of the track under the vertical position `clientY`, or null. */
function layerAt(clientX, clientY) {
  const lane = document.elementFromPoint(clientX, clientY)?.closest(".row.track .lane, .row.track");
  const layer = (lane?.matches(".lane") ? lane : lane?.querySelector(".lane"))?.dataset.layer;
  return layer === undefined ? null : Number(layer);
}

/** Drag `edge` ("move", "in" or "out") of clip `index` (and, moving, of every selected clip). */
function startDrag(event, edge, index) {
  event.stopPropagation(); // a drag on a clip edits it instead of scrubbing
  const additive = event.shiftKey || event.ctrlKey || event.metaKey;
  if (additive) {
    if (selected.has(index)) selected.delete(index);
    else selected.add(index);
    markSelected();
    drawClipPanel();
    return;
  }
  if (!selected.has(index) || edge !== "move") {
    selected.clear();
    selected.add(index);
  }
  markSelected();
  // Listened to on the window, not with pointer capture on the bar: moving
  // the bar into another track's lane takes it out of the document for a
  // moment, which ends its pointer capture.
  const { clips } = page.state.timeline;
  const moving = edge === "move" ? [...selected] : [index];
  const bars = new Map(moving.map((i) => [i, document.querySelector(`.clip-body[data-index="${i}"]`)]));
  const start = new Map(moving.map((i) => [i, { left: parseFloat(bars.get(i).style.left), width: parseFloat(bars.get(i).style.width) }]));
  // The segment bars under a moved clip move with it.
  const segments = edge === "move" ? moving.flatMap((i) => [...document.querySelectorAll(`.segment[data-clip="${i}"]`)].map((segment) => [segment, parseFloat(segment.style.left)])) : [];
  const secondsPerPixel = page.state.span / page.elements.ruler.getBoundingClientRect().width;
  const clip = clips[index];
  const grab = { x: event.clientX, delta: 0, layers: 0, moved: false };
  // What the drag can snap: the dragged edges, onto the edges of every clip that stays.
  const dragEdges =
    edge === "move" ? moving.flatMap((i) => [clips[i].start, clips[i].end].filter((time) => time !== null)) : [edge === "in" ? clip.start : clip.end];
  const targets = [
    0,
    page.state.time,
    ...page.state.timeline.events.map((event) => event.time).filter((time) => time !== null),
    ...clips.flatMap((other, i) => (edge === "move" ? moving.includes(i) : i === index) ? [] : [other.start, other.end].filter((time) => time !== null)),
  ];
  const send = liveSender("/api/clip", () => {
    if (edge === "move") page.refreshFrame();
    else page.showTime(edge === "in" ? clip.start + grab.delta : clip.end + grab.delta);
  });
  page.state.dragging = true;
  const onMove = (move) => {
    const raw = (move.clientX - grab.x) * secondsPerPixel;
    grab.moved ||= Math.abs(move.clientX - grab.x) > 2 || Math.abs(move.clientY - event.clientY) > 8;
    if (!grab.moved) return;
    // The dragged end lands on a frame.
    const from = edge === "out" ? clip.end : clip.start;
    let delta = onFrame(from + raw) - from;
    const keep = (value) => (edge === "move" ? Math.max(value, -Math.min(...moving.map((i) => clips[i].start))) : clamped(clip, edge, value));
    delta = keep(delta);
    const snap = move.altKey ? null : snapTo(delta, dragEdges, targets, SNAP_PIXELS * secondsPerPixel);
    if (snap) delta = keep(delta + snap.shift);
    showSnap(snap ? snap.target : null);
    let layers = grab.layers;
    if (edge === "move") {
      const over = layerAt(move.clientX, move.clientY);
      if (over !== null) layers = Math.max(over - clip.layer, -Math.min(...moving.map((i) => clips[i].layer)));
    }
    if (delta === grab.delta && layers === grab.layers) return;
    grab.delta = delta;
    grab.layers = layers;
    const shift = page.percent(delta);
    for (const i of moving) {
      const bar = bars.get(i);
      const { left, width } = start.get(i);
      if (edge === "move" || edge === "in") bar.style.left = `${left + shift}%`;
      if (edge === "in") bar.style.width = `${width - shift}%`;
      if (edge === "out") bar.style.width = `${width + shift}%`;
      const lane = document.querySelector(`.lane[data-layer="${clips[i].layer + layers}"]`);
      if (lane && bar.parentElement !== lane) lane.append(bar);
    }
    for (const [segment, left] of segments) segment.style.left = `${left + shift}%`;
    send({
      clips: moving.map((i) => ({
        index: i,
        ...dragChanges(clips[i], edge, delta),
        ...(edge === "move" ? { layer: clips[i].layer + layers } : {}),
      })),
    });
  };
  const onUp = async () => {
    removeEventListener("pointermove", onMove);
    removeEventListener("pointerup", onUp);
    showSnap(null);
    await send.idle();
    page.state.dragging = false;
    drawClipPanel();
    if (!grab.moved) page.redraw();
  };
  addEventListener("pointermove", onMove);
  addEventListener("pointerup", onUp);
}

/** The shift that puts one of `edges` (moved by `delta`) onto the nearest
 * of `targets` within `reach` seconds, and that target; or null. */
function snapTo(delta, edges, targets, reach) {
  let best = null;
  for (const at of edges) {
    for (const target of targets) {
      const shift = target - (at + delta);
      if (Math.abs(shift) <= reach && (!best || Math.abs(shift) < Math.abs(best.shift))) best = { shift, target };
    }
  }
  return best;
}

/** The line where a drag snapped, or none. */
function showSnap(time) {
  let line = document.getElementById("snap-line");
  if (!line) {
    line = document.createElement("div");
    line.id = "snap-line";
    document.getElementById("overlay").append(line);
  }
  line.hidden = time === null;
  if (time !== null) line.style.left = `${page.percent(time)}%`;
}

/** `time` on the nearest frame (past the timeline's end too: a clip may be moved there). */
function onFrame(time) {
  const { fps } = page.state.timeline;
  return Math.round(time * fps) / fps;
}

/** The clip's playing time range shown, from its cut and the scene's length. */
function playingRange(clip) {
  const speed = clip.entry.speed;
  const cutIn = clip.entry.cut?.[0] ?? 0;
  const cutOut = clip.entry.cut?.[1] ?? clip.length;
  return { cutIn, cutOut, speed, length: clip.length };
}

/** `delta` seconds of cutting kept within the scene (at least a frame stays). */
function clamped(clip, edge, delta) {
  const { cutIn, cutOut, speed, length } = playingRange(clip);
  const frame = 1 / page.state.timeline.fps;
  if (edge === "in") {
    const earliest = Math.max(-clip.start, -cutIn / speed);
    const latest = (cutOut - cutIn) / speed - frame;
    return Math.min(Math.max(delta, earliest), latest);
  }
  return Math.min(Math.max(delta, -((cutOut - cutIn) / speed - frame)), (length - cutOut) / speed);
}

/** The clip edit a drag of `delta` seconds on `edge` makes. */
function dragChanges(clip, edge, delta) {
  const { cutIn, cutOut, speed, length } = playingRange(clip);
  const round = (value) => Math.round(value * 1e6) / 1e6;
  if (edge === "move") return { start: round(clip.start + delta) };
  const cut = edge === "in" ? [round(cutIn + delta * speed), clip.entry.cut?.[1] ?? null] : [clip.entry.cut?.[0] ?? null, round(cutOut + delta * speed)];
  if (cut[0] !== null && cut[0] <= 1e-9) cut[0] = null; // back at the scene's start: no cut there
  if (cut[1] !== null && cut[1] >= length - 1e-9) cut[1] = null; // or at its end
  return edge === "in" ? { start: round(clip.start + delta), cut } : { cut };
}

// -- the clip panel ---------------------------------------------------------------

function drawClipPanel() {
  panels.clip.replaceChildren();
  const heading = document.createElement("h2");
  const { clips } = page.state.timeline;
  panels.clip.append(heading);
  if (selected.size === 0) {
    heading.textContent = "Clip";
    panels.clip.append(paragraph("Click a clip to edit it. Drag a clip to move it, its handles to cut its start or end; drag on empty space to select several, Shift or Ctrl click to add one."));
    return;
  }
  if (selected.size > 1) {
    heading.textContent = `${selected.size} clips`;
    panels.clip.append(paragraph([...selected].map((i) => clips[i].name).join(", ")), paragraph("Drag one of them to move them all. Escape clears the selection."), actions());
    return;
  }
  const index = [...selected][0];
  const clip = clips[index];
  heading.textContent = `Clip ${clip.name}`;
  const entry = clip.entry;
  const follows = entry.start === null;
  panels.clip.append(
    field("Start", numberInput(clip.start, 0, null, 1 / page.state.timeline.fps, (value) => editClip(index, { start: value })), "s", follows ? "after the previous clip; typing a time fixes it" : ""),
    follows ? "" : button("Follow the previous clip", () => editClip(index, { start: null })),
    field("Layer", numberInput(entry.layer, null, null, 1, (value) => editClip(index, { layer: Math.round(value) })), "", "higher layers are drawn on top"),
    field("Cut in", numberInput(entry.cut?.[0] ?? null, 0, clip.length, 0.01, (value) => editClip(index, { cut: [value, entry.cut?.[1] ?? null] }), "start"), "s", `in the scene's playing time (${clip.length.toFixed(2)} s long)`),
    field("Cut out", numberInput(entry.cut?.[1] ?? null, 0, clip.length, 0.01, (value) => editClip(index, { cut: [entry.cut?.[0] ?? null, value] }), "end"), "s"),
    field("Speed", numberInput(entry.speed, 0.01, null, 0.05, (value) => editClip(index, { speed: value })), "×", "2 plays twice as fast, 0.5 is slow motion"),
  );
  panels.clip.append(actions());
  for (const event of clip.waits) {
    panels.clip.append(field(`Waits for ${event}`, button("Fire at the playhead", () => setEvent(event, page.state.time)), "", "the clip goes on at that time"));
  }
}

/** Buttons for what S and Ctrl+D do to the selection. */
function actions() {
  const row = document.createElement("div");
  row.className = "actions";
  row.append(button("Split at the playhead (S)", split), button("Duplicate (Ctrl+D)", duplicate));
  return row;
}

// -- events: pills on the tracks' events rows and the events panel ----------------------

/** Make an event's pill draggable (app.js calls this for every wait shown). */
export function eventPill(pill, wait, bar) {
  if (!page.state.timeline.editable) return;
  pill.addEventListener("pointerdown", (down) => {
    down.stopPropagation();
    pill.setPointerCapture(down.pointerId);
    const from = wait.from ?? wait.clip.start;
    let time = wait.time;
    const send = liveSender("/api/event", () => page.showTime(time));
    page.state.dragging = true;
    pill.classList.remove("never");
    bar.classList.remove("never");
    const onMove = (move) => {
      const at = onFrame(timeAtUnclamped(move.clientX));
      if (at === time) return;
      time = at;
      pill.style.left = `${page.percent(time)}%`;
      bar.style.width = `${page.percent(Math.max(0, time - from))}%`;
      send({ name: wait.event, time });
    };
    const onUp = async () => {
      pill.removeEventListener("pointermove", onMove);
      pill.removeEventListener("pointerup", onUp);
      await send.idle();
      page.state.dragging = false;
    };
    pill.addEventListener("pointermove", onMove);
    pill.addEventListener("pointerup", onUp);
  });
}

function drawEventsPanel() {
  panels.events.replaceChildren();
  const heading = document.createElement("h2");
  heading.textContent = "Events";
  panels.events.append(heading);
  const events = page.state.timeline.events ?? [];
  if (events.length === 0) {
    panels.events.append(paragraph("No scene waits for an event (wait_for)."));
    return;
  }
  for (const event of events) {
    const state = event.fixed ? "set in manimx.json" : event.time === null ? "never fires: its clips wait forever" : "emitted by a scene";
    const controls = [numberInput(event.time, 0, null, 1 / page.state.timeline.fps, (value) => setEvent(event.name, value), "unset"), button("At the playhead", () => setEvent(event.name, page.state.time))];
    if (event.fixed) controls.push(button("Remove", () => setEvent(event.name, null)));
    panels.events.append(field(event.name, controls, "s", state));
  }
}

// -- the settings dialog ------------------------------------------------------------

async function openSettings() {
  panels.settingsForm.replaceChildren(paragraph("Loading…"));
  panels.settings.showModal();
  const response = await fetch("/api/settings");
  const items = await response.json();
  panels.settingsForm.replaceChildren();
  if (!response.ok) {
    panels.settingsForm.append(paragraph(items.error));
    return;
  }
  for (const item of items) {
    if (item.section) {
      const heading = document.createElement(item.path.length === 1 ? "h2" : "h3");
      heading.textContent = item.path.join(".");
      heading.title = item.description;
      panels.settingsForm.append(heading);
      continue;
    }
    panels.settingsForm.append(settingField(item));
  }
}

function settingField(item) {
  const name = item.path.at(-1);
  const save = async (value, input) => {
    const ok = await post("/api/setting", { path: item.path, value });
    input.setCustomValidity(ok ? "" : panels.message.textContent);
    input.reportValidity();
  };
  let input;
  if (item.type === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = item.value;
    input.addEventListener("change", () => save(input.checked, input));
  } else if (item.choices) {
    input = document.createElement("select");
    for (const choice of item.choices) input.append(new Option(String(choice), String(choice), false, choice === item.value));
    input.addEventListener("change", () => save(item.type === "int" ? Number(input.value) : item.type === "float" ? Number(input.value) : input.value, input));
  } else if (item.type === "int" || item.type === "float") {
    input = numberInput(item.value, item.minimum, item.maximum, item.type === "int" ? 1 : "any", (value, element) => save(item.type === "int" ? Math.round(value) : value, element));
  } else {
    input = document.createElement("input");
    input.type = "text";
    input.value = item.value;
    input.addEventListener("change", () => save(input.value, input));
  }
  const parts = [input];
  if (item.color) {
    const swatch = document.createElement("input");
    swatch.type = "color";
    if (/^#[0-9a-f]{6}$/i.test(item.value)) swatch.value = item.value;
    swatch.addEventListener("change", () => {
      input.value = swatch.value;
      save(swatch.value, input);
    });
    parts.push(swatch);
  }
  const description = item.note ? `${item.description} (${item.note})` : item.description;
  return field(name, parts, item.unit ?? "", description);
}

// -- form parts ------------------------------------------------------------------------

/** A number field that calls `change(value, input)` when a valid number is entered (null when emptied and `empty` names what empty means). */
function numberInput(value, minimum, maximum, step, change, empty = null) {
  const input = document.createElement("input");
  input.type = "number";
  input.step = String(step);
  if (minimum !== null && minimum !== undefined) input.min = String(minimum);
  if (maximum !== null && maximum !== undefined) input.max = String(maximum);
  input.value = value === null || value === undefined ? "" : String(Number(value.toFixed?.(4) ?? value));
  if (empty) input.placeholder = empty;
  input.addEventListener("change", () => {
    if (input.value === "") {
      if (empty) change(null, input);
      return;
    }
    if (input.checkValidity()) change(Number(input.value), input);
    else input.reportValidity();
  });
  return input;
}

function field(label, controls, unit = "", hint = "") {
  const row = document.createElement("label");
  row.className = "field";
  const name = document.createElement("span");
  name.className = "field-name";
  name.textContent = label;
  const box = document.createElement("span");
  box.className = "field-controls";
  box.append(...[controls].flat());
  if (unit) {
    const units = document.createElement("span");
    units.className = "unit";
    units.textContent = unit;
    box.append(units);
  }
  row.append(name, box);
  if (hint) {
    const note = document.createElement("small");
    note.textContent = hint;
    row.append(note);
  }
  return row;
}

function button(text, click) {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = text;
  element.addEventListener("click", (event) => {
    event.preventDefault(); // inside a label: no second click on the field
    click();
  });
  return element;
}

function paragraph(text) {
  const element = document.createElement("p");
  element.className = "hint";
  element.textContent = text;
  return element;
}
