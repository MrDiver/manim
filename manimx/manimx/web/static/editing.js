// Editing the project from the preview page (manimx/web/editing.py): clips
// are moved by dragging their bars, cut by dragging their ends, and their
// start, layer, cut and speed typed in the clip panel; events are dragged on
// the ruler or set at the playhead in the events panel; the settings dialog
// changes manimx.json's settings.
//
// Every edit is one POST, sent when a drag is let go or a field is changed.
// The server saves manimx.json, the reloader rebuilds the timeline, and the
// page shows it when /api/changes answers (app.js), so nothing here keeps
// its own copy of the project. While a drag is under way, only the bar or
// marker being dragged moves.

/** Pixels from a clip bar's end where a drag cuts instead of moving it. */
const EDGE = 6;

let page = null; // what app.js shares: state, elements and its helpers
let selected = null; // index of the clip shown in the clip panel

const panels = {
  editor: document.getElementById("editor"),
  clip: document.getElementById("clip-editor"),
  events: document.getElementById("events-editor"),
  message: document.getElementById("edit-message"),
  settingsButton: document.getElementById("settings-button"),
  settings: document.getElementById("settings"),
  settingsForm: document.getElementById("settings-form"),
};

export function initEditing(shared) {
  page = shared;
  panels.settingsButton.addEventListener("click", openSettings);
  panels.settings.addEventListener("close", () => panels.settingsForm.replaceChildren());
}

/** Called by app.js whenever the timeline was (re)loaded. */
export function timelineLoaded() {
  const { editable } = page.state.timeline;
  panels.editor.hidden = !editable;
  panels.settingsButton.hidden = !editable;
  if (!editable) return;
  if (panels.message.dataset.waiting) {
    delete panels.message.dataset.waiting;
    say(page.state.timeline.error ? "Saved, but the project does not build: see above." : "Saved to manimx.json.", Boolean(page.state.timeline.error));
  }
  if (selected !== null && !page.state.timeline.clips[selected]) selected = null;
  drawClipPanel();
  drawEventsPanel();
}

// -- sending edits ------------------------------------------------------------

async function post(route, body) {
  say("Saving…");
  try {
    const response = await fetch(route, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.error);
    say("Saved to manimx.json; rebuilding…");
    panels.message.dataset.waiting = "yes"; // until the rebuilt timeline arrives
    return true;
  } catch (error) {
    say(error.message, true);
    page.redraw(); // put dragged bars back
    return false;
  }
}

function say(text, problem = false) {
  panels.message.textContent = text;
  panels.message.classList.toggle("problem", problem);
}

const editClip = (index, changes) => post("/api/clip", { index, ...changes });
const setEvent = (name, time) => post("/api/event", { name, time });

// -- clip bars ------------------------------------------------------------------

/** Make a clip's bar draggable (app.js calls this for every clip bar). */
export function clipBar(body, clip, index) {
  if (!page.state.timeline.editable || !clip.entry) return;
  body.classList.add("editable");
  body.classList.toggle("selected", index === selected);
  body.addEventListener("pointermove", (event) => {
    if (!body.hasPointerCapture(event.pointerId)) body.style.cursor = cursorFor(edgeAt(body, event.clientX, clip));
  });
  body.addEventListener("pointerdown", (event) => {
    event.stopPropagation(); // a drag on a clip edits it instead of scrubbing
    body.setPointerCapture(event.pointerId);
    const grab = { edge: edgeAt(body, event.clientX, clip), x: event.clientX, moved: false };
    const secondsPerPixel = page.state.timeline.duration / page.elements.ruler.getBoundingClientRect().width;
    const left = parseFloat(body.style.left);
    const width = parseFloat(body.style.width);
    const onMove = (move) => {
      const delta = (move.clientX - grab.x) * secondsPerPixel;
      grab.moved ||= Math.abs(move.clientX - grab.x) > 2;
      // The dragged end lands on a frame.
      const from = grab.edge === "out" ? clip.end : clip.start;
      grab.delta = clamped(clip, grab.edge, onFrame(from + delta) - from);
      const shift = page.percent(grab.delta);
      if (grab.edge === "move") body.style.left = `${left + shift}%`;
      if (grab.edge === "in") {
        body.style.left = `${left + shift}%`;
        body.style.width = `${width - shift}%`;
      }
      if (grab.edge === "out") body.style.width = `${width + shift}%`;
    };
    const onUp = () => {
      body.removeEventListener("pointermove", onMove);
      body.removeEventListener("pointerup", onUp);
      if (!grab.moved || !grab.delta) {
        select(index);
        return;
      }
      select(index);
      editClip(index, dragChanges(clip, grab.edge, grab.delta));
    };
    body.addEventListener("pointermove", onMove);
    body.addEventListener("pointerup", onUp);
  });
}

/** Which part of the bar is under `clientX`: its in or out end, or the middle. */
function edgeAt(body, clientX, clip) {
  const box = body.getBoundingClientRect();
  if (clientX - box.left < EDGE) return "in";
  if (box.right - clientX < EDGE && clip.end !== null) return "out"; // a waiting clip has no end to drag
  return "move";
}

/** `time` on the nearest frame (past the timeline's end too: a clip may be moved there). */
function onFrame(time) {
  const { fps } = page.state.timeline;
  return Math.round(time * fps) / fps;
}

function cursorFor(edge) {
  return edge === "move" ? "grab" : "col-resize";
}

/** The clip's playing time range shown, from its cut and duration. */
function playingRange(clip) {
  const speed = clip.entry.speed;
  const cutIn = clip.entry.cut?.[0] ?? 0;
  const length = clip.end === null ? null : (clip.end - clip.start) * speed;
  const cutOut = clip.entry.cut?.[1] ?? (length === null ? null : cutIn + length);
  return { cutIn, cutOut, speed };
}

/** `delta` seconds kept within what the clip allows. */
function clamped(clip, edge, delta) {
  const { cutIn, cutOut, speed } = playingRange(clip);
  const frame = 1 / page.state.timeline.fps;
  if (edge === "move") return Math.max(delta, -clip.start);
  if (edge === "in") {
    const earliest = Math.max(-clip.start, -cutIn / speed);
    const latest = cutOut === null ? Infinity : (cutOut - cutIn) / speed - frame;
    return Math.min(Math.max(delta, earliest), latest);
  }
  return Math.max(delta, -((cutOut - cutIn) / speed - frame));
}

/** The clip edit a drag of `delta` seconds on `edge` makes. */
function dragChanges(clip, edge, delta) {
  const { cutIn, cutOut, speed } = playingRange(clip);
  const round = (value) => Math.round(value * 1e6) / 1e6;
  if (edge === "move") return { start: round(clip.start + delta) };
  if (edge === "in") return { start: round(clip.start + delta), cut: [round(cutIn + delta * speed) || null, clip.entry.cut?.[1] ?? null] };
  return { cut: [clip.entry.cut?.[0] ?? null, round(cutOut + delta * speed)] };
}

function select(index) {
  selected = index;
  for (const bar of document.querySelectorAll(".clip-body.editable")) bar.classList.remove("selected");
  page.redraw();
  drawClipPanel();
}

// -- the clip panel ---------------------------------------------------------------

function drawClipPanel() {
  panels.clip.replaceChildren();
  const heading = document.createElement("h2");
  const clip = selected === null ? null : page.state.timeline.clips[selected];
  heading.textContent = clip ? `Clip ${clip.name}` : "Clip";
  panels.clip.append(heading);
  if (!clip) {
    panels.clip.append(paragraph("Click a clip to edit it; drag it to move it, drag its ends to cut it."));
    return;
  }
  const entry = clip.entry;
  const index = selected;
  const follows = entry.start === null;
  panels.clip.append(
    field("Start", numberInput(clip.start, 0, null, 1 / page.state.timeline.fps, (value) => editClip(index, { start: value })), "s", follows ? "after the previous clip; typing a time fixes it" : ""),
    follows ? "" : button("Follow the previous clip", () => editClip(index, { start: null })),
    field("Layer", numberInput(entry.layer, null, null, 1, (value) => editClip(index, { layer: Math.round(value) })), "", "higher layers are drawn on top"),
    field("Cut in", numberInput(entry.cut?.[0] ?? null, 0, null, 0.01, (value) => editClip(index, { cut: [value, entry.cut?.[1] ?? null] }), "start"), "s", "in the scene's playing time"),
    field("Cut out", numberInput(entry.cut?.[1] ?? null, 0, null, 0.01, (value) => editClip(index, { cut: [entry.cut?.[0] ?? null, value] }), "end"), "s"),
    field("Speed", numberInput(entry.speed, 0.01, null, 0.05, (value) => editClip(index, { speed: value })), "×", "2 plays twice as fast, 0.5 is slow motion"),
  );
  for (const event of clip.waits) {
    panels.clip.append(field(`Waits for ${event}`, button("Fire at the playhead", () => setEvent(event, page.state.time)), "", "the clip goes on at that time"));
  }
}

// -- events: markers on the ruler and the events panel ----------------------------------

/** Draw a marker per event with a time on the ruler (app.js calls this). */
export function eventMarkers(ruler) {
  const { editable, events } = page.state.timeline;
  for (const event of events ?? []) {
    if (event.time === null) continue;
    const marker = document.createElement("div");
    marker.className = `event${event.fixed ? " fixed" : ""}`;
    marker.style.left = `${page.percent(event.time)}%`;
    marker.title = `${event.name} at ${event.time.toFixed(2)} s${event.fixed ? (editable ? " (drag to move)" : "") : " (emitted by a scene)"}`;
    marker.textContent = event.name;
    if (editable && event.fixed) dragMarker(marker, event);
    ruler.append(marker);
  }
}

function dragMarker(marker, event) {
  marker.addEventListener("pointerdown", (down) => {
    down.stopPropagation();
    marker.setPointerCapture(down.pointerId);
    let time = event.time;
    const onMove = (move) => {
      time = page.timeAt(move.clientX);
      marker.style.left = `${page.percent(time)}%`;
    };
    const onUp = () => {
      marker.removeEventListener("pointermove", onMove);
      marker.removeEventListener("pointerup", onUp);
      if (time !== event.time) setEvent(event.name, time);
    };
    marker.addEventListener("pointermove", onMove);
    marker.addEventListener("pointerup", onUp);
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
