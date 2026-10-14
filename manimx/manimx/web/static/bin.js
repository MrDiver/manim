// The project bin, left of the frame: what the project has to put on the
// timeline. Scenes (editing.js draws them), Scripts and Sounds; each is
// dragged onto a track.
//
// Scripts: every script of the project (scripts/<name>.txt), and below it
// its parts, the stretches between "---" lines: what is recorded in one
// go. A part shows its first words, how long it lasts and whether it is
// recorded; drag it onto a script track to place it (already placed: it
// moves there). Clicking a script opens it in the Script tab; "+ New"
// starts a new one there.
//
// Sounds: the sound files in assets/audio/; drag one onto an audio track.
// "+ Import" adds files from the computer (they are copied into the
// project); dropping files from the desktop onto a track does too.

import { openScript } from "./scriptview.js";
import { say } from "./editing.js";

export const PART_TYPE = "application/x-manimx-part";
export const SOUND_TYPE = "application/x-manimx-sound";

let page = null;
let scripts = []; // /api/scripts
let sounds = []; // /api/sounds

const elements = {
  scripts: document.getElementById("script-list"),
  sounds: document.getElementById("sound-list"),
  newScript: document.getElementById("new-script"),
  importSound: document.getElementById("import-sound"),
  soundFile: document.getElementById("sound-file"),
};

export function initBin(shared) {
  page = shared;
  elements.newScript.addEventListener("click", () => openScript(null));
  elements.importSound.addEventListener("click", () => elements.soundFile.click());
  elements.soundFile.addEventListener("change", async () => {
    for (const file of elements.soundFile.files) await uploadSound(file, file.name);
    elements.soundFile.value = "";
    refreshBin();
  });
}

/** The scripts as last fetched (audiotracks.js and scriptview.js read them). */
export const projectScripts = () => scripts;

/** Part `part` of script `name`, or undefined. */
export const scriptPart = (name, part) => scripts.find((script) => script.name === name)?.parts?.[part];

/** Fetch the scripts and sounds again and draw them (after every change of the timeline). */
export async function refreshBin() {
  if (!page.state.timeline?.editable) return;
  const [scriptAnswer, soundAnswer] = await Promise.all([fetch("/api/scripts"), fetch("/api/sounds")]);
  if (scriptAnswer.ok) scripts = await scriptAnswer.json();
  if (soundAnswer.ok) sounds = await soundAnswer.json();
  drawScripts();
  drawSounds();
  document.dispatchEvent(new CustomEvent("manimx:scripts"));
}

/** Upload a sound file into assets/audio/; its path in the project, or null. */
export async function uploadSound(blob, name) {
  say(`Copying ${name} into the project…`);
  try {
    const response = await fetch(`/api/asset?name=${encodeURIComponent(name)}`, { method: "POST", body: blob });
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.error);
    say(`${name} is in the project's sounds.`);
    return answer.file;
  } catch (error) {
    say(`${name}: ${error.message}`, true);
    return null;
  }
}

function drawScripts() {
  elements.scripts.replaceChildren();
  if (!scripts.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "No script yet. Write what is said, then drag its parts onto a script track.";
    elements.scripts.append(empty);
    return;
  }
  for (const script of scripts) {
    const block = document.createElement("div");
    block.className = "bin-script";
    const head = document.createElement("button");
    head.type = "button";
    head.className = "bin-script-name";
    head.textContent = script.name;
    head.title = "Open in the Script tab";
    head.addEventListener("click", () => openScript(script.name));
    block.append(head);
    if (script.error) {
      const problem = document.createElement("p");
      problem.className = "hint problem";
      problem.textContent = script.error;
      block.append(problem);
    }
    const list = document.createElement("ul");
    list.className = "bin-list parts";
    (script.parts ?? []).forEach((part, index) => list.append(partItem(script, part, index)));
    block.append(list);
    elements.scripts.append(block);
  }
}

function partItem(script, part, index) {
  const item = document.createElement("li");
  item.draggable = true;
  item.className = `part${part.take ? " recorded" : ""}${part.clip !== null ? " placed" : ""}`;
  const title = document.createElement("span");
  title.className = "part-title";
  title.textContent = `${index + 1}. ${part.title || "(empty)"}`;
  const meta = document.createElement("span");
  meta.className = "count";
  meta.textContent = `${part.take ? "●" : "○"} ${part.length.toFixed(1)} s`;
  meta.title = part.take ? "Recorded" : "Not recorded yet: timed as if read out";
  item.append(title, meta);
  item.title = part.clip !== null ? "On the timeline: drag to move it" : "Drag onto a script track to place it";
  item.addEventListener("dragstart", (event) => {
    event.dataTransfer.setData(PART_TYPE, JSON.stringify({ script: script.name, part: index }));
    event.dataTransfer.effectAllowed = "copyMove";
  });
  item.addEventListener("dblclick", () => openScript(script.name));
  return item;
}

function drawSounds() {
  elements.sounds.replaceChildren();
  if (!sounds.length) {
    const empty = document.createElement("li");
    empty.className = "hint";
    empty.textContent = "Import music or effects, or drop sound files onto an audio track.";
    elements.sounds.append(empty);
    return;
  }
  for (const sound of sounds) {
    const item = document.createElement("li");
    item.draggable = true;
    const name = document.createElement("span");
    name.className = "part-title";
    name.textContent = sound.name;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = sound.clips ? `×${sound.clips}` : "";
    item.append(name, count);
    item.title = `${sound.file}: drag onto an audio track`;
    item.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData(SOUND_TYPE, sound.file);
      event.dataTransfer.effectAllowed = "copy";
    });
    elements.sounds.append(item);
  }
}
