// Audio tracks: sound files on the timeline, outside any scene (a
// voice-over, music, a recording), below the scenes' tracks.
//
// Drop sound files from the desktop onto an audio track (or "New audio
// track") to put them there at the drop time: each is uploaded into the
// project's assets/audio/ (POST /api/asset) and placed (/api/audio/add).
// Record (the microphone button) plays the timeline from the playhead
// and records the microphone meanwhile, so a voice-over is spoken to the
// picture; stopping puts the recording on the first audio track free for
// it, where the recording started.
//
// An audio clip's bar shows its waveform (/api/waveform, the peaks of the
// part it plays). Like scene clips: click selects (Shift or Ctrl adds),
// drag moves it (up or down to another audio track), its ends cut it,
// drags snap (Alt: freely), Delete removes, every edit is undone with
// Ctrl+Z. The inspector shows the selected audio clip's start and volume.

import { button, field, liveSender, numberInput, onFrame, paragraph, post, say, showSnap, snapTo, timeAtUnclamped } from "./editing.js";

const SNAP_PIXELS = 8;
const selected = new Set(); // indices of the selected audio clips
const waveforms = new Map(); // request URL: peaks
let page = null;
let recording = null; // { recorder, start, chunks } while recording

const panel = document.getElementById("audio-editor");
const recordButton = document.getElementById("record");

export function initAudio(shared) {
  page = shared;
  recordButton.addEventListener("click", () => (recording ? stopRecording() : startRecording()));
  document.addEventListener("manimx:select", (event) => {
    if (event.detail !== "audio" && selected.size) {
      selected.clear();
      markSelected();
      drawPanel();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input, select, textarea, dialog") || !page.state.timeline?.editable) return;
    if ((event.key === "Delete" || event.key === "Backspace") && selected.size) {
      event.preventDefault();
      const indices = [...selected];
      selected.clear();
      post("/api/audio/remove", { indices });
    }
    if (event.key === "Escape" && selected.size) {
      selected.clear();
      markSelected();
      drawPanel();
    }
  });
}

/** Called by app.js after the scenes' tracks: the audio tracks' rows. */
export function drawAudioTracks() {
  const { audio = [], editable } = page.state.timeline;
  for (const index of [...selected]) if (!audio[index]) selected.delete(index);
  recordButton.hidden = !editable;
  if (!editable && !audio.length) return;
  const tracks = [...new Set(audio.map((clip) => clip.track))].sort((a, b) => a - b);
  const free = tracks.length ? tracks[tracks.length - 1] + 1 : 0;
  for (const track of tracks) {
    const { lane } = page.addRow(`audio:${track}`, 0, `Audio ${track + 1}`, "track audio-track", false);
    audioLane(lane, track);
    audio.forEach((clip, index) => {
      if (clip.track === track) lane.append(audioBar(clip, index));
    });
  }
  if (editable) {
    const { row, lane } = page.addRow(`audio:${free}`, 0, "New audio track", "track audio-track new-track", false);
    row.title = "Drop sound files here to put them on a new audio track";
    audioLane(lane, free);
  }
  drawPanel();
}

// -- dropping and recording -------------------------------------------------------

function audioLane(lane, track) {
  lane.dataset.audioTrack = track;
  if (!page.state.timeline.editable) return;
  const row = lane.parentElement;
  lane.addEventListener("dragover", (event) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    row.classList.add("drop");
  });
  lane.addEventListener("dragleave", () => row.classList.remove("drop"));
  lane.addEventListener("drop", async (event) => {
    row.classList.remove("drop");
    const files = [...event.dataTransfer.files];
    if (!files.length) return;
    event.preventDefault();
    let start = onFrame(timeAtUnclamped(event.clientX));
    for (const file of files) {
      const length = await placeFile(file, file.name, start, track);
      if (length === null) break;
      start = onFrame(start + length); // several files: one after the other
    }
  });
  lane.addEventListener("pointerdown", (event) => {
    if (event.target !== lane) return;
    if (selected.size) {
      selected.clear();
      markSelected();
      drawPanel();
    }
  });
}

/** Upload `blob` as `name` and put it on `track` at `start`; its length in seconds, or null if that failed. */
async function placeFile(blob, name, start, track) {
  say(`Uploading ${name}…`);
  try {
    const response = await fetch(`/api/asset?name=${encodeURIComponent(name)}`, { method: "POST", body: blob });
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.error);
    if (!(await post("/api/audio/add", { file: answer.file, start, track }))) return null;
    const timeline = await (await fetch("/api/timeline")).json();
    const added = timeline.audio.findLast((clip) => clip.file === answer.file);
    return added ? added.end - added.start : 0;
  } catch (error) {
    say(`${name}: ${error.message}`, true);
    return null;
  }
}

async function startRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
  } catch (error) {
    say(`No microphone: ${error.message}`, true);
    return;
  }
  const recorder = new MediaRecorder(stream);
  recording = { recorder, start: page.state.time, chunks: [] };
  recorder.addEventListener("dataavailable", (event) => recording?.chunks.push(event.data));
  recorder.start();
  recordButton.setAttribute("aria-pressed", "true");
  say("Recording: speak to the picture; press the record button again to stop.");
  page.play();
}

async function stopRecording() {
  const { recorder, start, chunks } = recording;
  const stopped = new Promise((resolve) => recorder.addEventListener("stop", resolve, { once: true }));
  recorder.stop();
  page.pause();
  await stopped;
  for (const track of recorder.stream.getTracks()) track.stop();
  recording = null;
  recordButton.setAttribute("aria-pressed", "false");
  const blob = new Blob(chunks, { type: recorder.mimeType });
  const suffix = recorder.mimeType.includes("ogg") ? "ogg" : recorder.mimeType.includes("mp4") ? "m4a" : "webm";
  // The first audio track with room for it from where it started (its length is not known before it is decoded).
  const audio = page.state.timeline.audio ?? [];
  const tracks = [...new Set(audio.map((clip) => clip.track))].sort((a, b) => a - b);
  const track = tracks.find((t) => !audio.some((clip) => clip.track === t && clip.end > start)) ?? (tracks.length ? tracks[tracks.length - 1] + 1 : 0);
  await placeFile(blob, `recording.${suffix}`, onFrame(start), track);
}

// -- bars: waveform, selecting, dragging -------------------------------------------------

function audioBar(clip, index) {
  const bar = document.createElement("div");
  bar.className = "clip-body audio-body";
  bar.dataset.audio = index;
  bar.style.left = `${page.percent(clip.start)}%`;
  bar.style.width = `${page.percent(clip.end - clip.start)}%`;
  bar.title = `${clip.file}: ${clip.start.toFixed(2)} – ${clip.end.toFixed(2)} s`;
  bar.classList.toggle("selected", selected.has(index));
  const canvas = document.createElement("canvas");
  canvas.className = "waveform";
  bar.append(canvas);
  const name = document.createElement("span");
  name.className = "clip-name";
  name.textContent = clip.name;
  bar.append(name);
  drawWaveform(canvas, clip, index, (page.percent(clip.end - clip.start) / 100) * page.elements.ruler.offsetWidth);
  if (page.state.timeline.editable) {
    bar.classList.add("editable");
    for (const edge of ["in", "out"]) {
      const handle = document.createElement("div");
      handle.className = `handle ${edge}`;
      handle.title = edge === "in" ? "Drag to cut the start" : "Drag to cut the end";
      handle.addEventListener("pointerdown", (event) => startDrag(event, edge, index));
      bar.append(handle);
    }
    bar.addEventListener("pointerdown", (event) => startDrag(event, "move", index));
  }
  return bar;
}

async function drawWaveform(canvas, clip, index, pixels) {
  const buckets = Math.max(1, Math.min(4000, Math.round(pixels / 2)));
  const from = clip.cut?.[0] ?? 0;
  const to = clip.cut?.[1] ?? clip.length;
  const url = `/api/waveform?index=${index}&from=${from}&to=${to}&n=${buckets}&file=${encodeURIComponent(clip.file)}`;
  let peaks = waveforms.get(url);
  if (!peaks) {
    try {
      const response = await fetch(url);
      if (!response.ok) return;
      peaks = await response.json();
      waveforms.set(url, peaks);
    } catch {
      return;
    }
  }
  canvas.width = buckets;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  context.fillStyle = "rgb(120 220 160 / 0.8)";
  peaks.forEach(([low, high], x) => {
    const top = 32 - high * 32 * clip.volume;
    context.fillRect(x, top, 1, Math.max(1, (high - low) * 32 * clip.volume));
  });
}

function markSelected() {
  for (const bar of document.querySelectorAll(".audio-body")) bar.classList.toggle("selected", selected.has(Number(bar.dataset.audio)));
}

/** The audio track under the vertical position `clientY`, or null. */
function trackAt(clientX, clientY) {
  const lane = document.elementFromPoint(clientX, clientY)?.closest(".row.audio-track")?.querySelector(".lane");
  return lane ? Number(lane.dataset.audioTrack) : null;
}

/** Drag `edge` ("move", "in" or "out") of audio clip `index` (moving: every selected one). */
function startDrag(event, edge, index) {
  event.stopPropagation();
  const additive = event.shiftKey || event.ctrlKey || event.metaKey;
  if (additive) {
    if (selected.has(index)) selected.delete(index);
    else selected.add(index);
  } else if (!selected.has(index) || edge !== "move") {
    selected.clear();
    selected.add(index);
  }
  document.dispatchEvent(new CustomEvent("manimx:select", { detail: "audio" }));
  markSelected();
  drawPanel();
  if (additive) return;
  const { audio, clips } = page.state.timeline;
  const clip = audio[index];
  const moving = edge === "move" ? [...selected] : [index];
  const bars = new Map(moving.map((i) => [i, document.querySelector(`.audio-body[data-audio="${i}"]`)]));
  const start = new Map(moving.map((i) => [i, { left: parseFloat(bars.get(i).style.left), width: parseFloat(bars.get(i).style.width) }]));
  const secondsPerPixel = page.state.span / page.elements.ruler.getBoundingClientRect().width;
  const edges = edge === "move" ? moving.flatMap((i) => [audio[i].start, audio[i].end]) : [edge === "in" ? clip.start : clip.end];
  const targets = [
    0,
    page.state.time,
    ...clips.flatMap((other) => [other.start, other.end].filter((time) => time !== null)),
    ...audio.flatMap((other, i) => (moving.includes(i) ? [] : [other.start, other.end])),
  ];
  const grab = { delta: 0, tracks: 0, moved: false };
  const send = liveSender("/api/audio/edit", () => {
    if (edge !== "move") page.showTime(edge === "in" ? clip.start + grab.delta : clip.end + grab.delta);
  });
  page.state.dragging = true;
  const keep = (delta) => {
    const cutIn = clip.cut?.[0] ?? 0;
    const cutOut = clip.cut?.[1] ?? clip.length;
    if (edge === "move") return Math.max(delta, -Math.min(...moving.map((i) => audio[i].start)));
    if (edge === "in") return Math.min(Math.max(delta, -cutIn, -clip.start), cutOut - cutIn - 0.01);
    return Math.min(Math.max(delta, -(cutOut - cutIn - 0.01)), clip.length - cutOut);
  };
  const onMove = (move) => {
    grab.moved ||= Math.abs(move.clientX - event.clientX) > 2 || Math.abs(move.clientY - event.clientY) > 8;
    if (!grab.moved) return;
    let delta = keep(onFrame((move.clientX - event.clientX) * secondsPerPixel));
    const snap = move.altKey ? null : snapTo(delta, edges, targets, SNAP_PIXELS * secondsPerPixel);
    if (snap) delta = keep(delta + snap.shift);
    showSnap(snap ? snap.target : null);
    let tracks = grab.tracks;
    if (edge === "move") {
      const over = trackAt(move.clientX, move.clientY);
      if (over !== null) tracks = Math.max(over - clip.track, -Math.min(...moving.map((i) => audio[i].track)));
    }
    if (delta === grab.delta && tracks === grab.tracks) return;
    grab.delta = delta;
    grab.tracks = tracks;
    const shift = page.percent(delta);
    for (const i of moving) {
      const bar = bars.get(i);
      const { left, width } = start.get(i);
      if (edge !== "out") bar.style.left = `${left + shift}%`;
      if (edge === "in") bar.style.width = `${width - shift}%`;
      if (edge === "out") bar.style.width = `${width + shift}%`;
      const lane = document.querySelector(`.lane[data-audio-track="${audio[i].track + tracks}"]`);
      if (lane && bar.parentElement !== lane) lane.append(bar);
    }
    const round = (value) => Math.round(value * 1e6) / 1e6;
    send({
      clips: moving.map((i) => {
        const other = audio[i];
        if (edge === "move") return { index: i, start: round(other.start + delta), track: other.track + tracks };
        const cutIn = other.cut?.[0] ?? 0;
        const cutOut = other.cut?.[1] ?? other.length;
        const cut = edge === "in" ? [round(cutIn + delta), other.cut?.[1] ?? null] : [other.cut?.[0] ?? null, round(cutOut + delta)];
        if (cut[0] !== null && cut[0] <= 1e-9) cut[0] = null;
        if (cut[1] !== null && cut[1] >= other.length - 1e-9) cut[1] = null;
        return edge === "in" ? { index: i, start: round(other.start + delta), cut } : { index: i, cut };
      }),
    });
  };
  const onUp = async () => {
    removeEventListener("pointermove", onMove);
    removeEventListener("pointerup", onUp);
    showSnap(null);
    await send.idle();
    page.state.dragging = false;
    if (!grab.moved) page.redraw();
  };
  addEventListener("pointermove", onMove);
  addEventListener("pointerup", onUp);
}

// -- the inspector ------------------------------------------------------------------

function drawPanel() {
  panel.replaceChildren();
  if (!selected.size) return;
  const { audio } = page.state.timeline;
  const heading = document.createElement("h2");
  panel.append(heading);
  if (selected.size > 1) {
    heading.textContent = `${selected.size} audio clips`;
    panel.append(paragraph("Drag one of them to move them all; Delete removes them."));
    return;
  }
  const index = [...selected][0];
  const clip = audio[index];
  const edit = (changes) => post("/api/audio/edit", { index, ...changes });
  heading.textContent = `Audio ${clip.name}`;
  panel.append(
    paragraph(clip.file),
    field("Start", numberInput(clip.start, 0, null, 1 / page.state.timeline.fps, (value) => edit({ start: value })), "s"),
    field("Track", numberInput(clip.track + 1, 1, null, 1, (value) => edit({ track: Math.round(value) - 1 })), ""),
    field("Volume", numberInput(clip.volume, 0, 4, 0.05, (value) => edit({ volume: value })), "×", "1 is the file's own loudness"),
    button("Remove (Delete)", () => {
      selected.clear();
      post("/api/audio/remove", { indices: [index] });
    }),
  );
}
