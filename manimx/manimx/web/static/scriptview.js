// The Script tab: writing what is said, next to the Preview tab.
//
// Pick a script (or start a new one: give it a name, then write), and
// write in the big text box. A line "---" ends a part, what is recorded in
// one go; a line "[name]" names a section. The text is saved as you write
// (a moment after the last key; Ctrl+Z undoes it like any edit), and the
// words keep their times. Below the text, every part: how long it lasts,
// whether it is recorded and where it is on the timeline; "Place at the
// playhead" puts it on a free audio track there (or drag it from the
// bin). The timeline stays below the tab, so placed parts show while you
// write.

import { projectScripts, refreshBin } from "./bin.js";
import { button, post } from "./editing.js";
import { trackName } from "./trackheads.js";

const SAVE_DELAY = 700; // ms after the last key
let page = null;
let current = null; // the name of the script shown
let writingNew = false; // a new script is being written (current is null then)
let timer = null;

const elements = {
  tabs: document.querySelector(".tabs"),
  preview: document.getElementById("tab-preview"),
  script: document.getElementById("tab-script"),
  stage: document.getElementById("stage"),
  view: document.getElementById("script-view"),
};

export function initScriptView(shared) {
  page = shared;
  elements.preview.addEventListener("click", () => showTab("preview"));
  elements.script.addEventListener("click", () => showTab("script"));
  document.addEventListener("manimx:scripts", () => {
    if (!elements.view.hidden) draw();
  });
}

/** Called when the timeline loaded: the tabs exist only where the project can be edited. */
export function scriptViewLoaded() {
  elements.tabs.hidden = !page.state.timeline.editable;
}

/** Show the script `name` in the Script tab (null: a new one). */
export function openScript(name) {
  current = name;
  writingNew = name === null;
  showTab("script");
}

function showTab(tab) {
  const script = tab === "script";
  elements.preview.setAttribute("aria-selected", String(!script));
  elements.script.setAttribute("aria-selected", String(script));
  elements.stage.hidden = script;
  elements.view.hidden = !script;
  if (script) {
    if (!writingNew) current ??= projectScripts()[0]?.name ?? null;
    writingNew = current === null;
    draw();
  }
}

function draw() {
  const scripts = projectScripts();
  const script = scripts.find((item) => item.name === current) ?? null;
  const text = elements.view.querySelector("textarea");
  if (text && document.activeElement === text && script) {
    drawParts(script); // writing: keep the text box as it is, only the parts follow
    return;
  }
  elements.view.replaceChildren();
  const head = document.createElement("div");
  head.className = "script-head";
  const picker = document.createElement("select");
  picker.title = "The script to write";
  for (const item of scripts) picker.append(new Option(item.name, item.name, false, item.name === current));
  picker.append(new Option("New script…", "", false, script === null));
  picker.addEventListener("change", () => {
    current = picker.value || null;
    writingNew = current === null;
    draw();
  });
  head.append(picker);
  let name = null;
  if (script === null) {
    name = document.createElement("input");
    name.className = "script-name";
    name.placeholder = "its name, e.g. narration";
    name.title = "The script's name: its words' events start with it (narration.square)";
    name.value = uniqueName(scripts);
    head.append(name);
  }
  const status = document.createElement("span");
  status.className = "script-status";
  head.append(status);
  if (script !== null) {
    const remove = button("Delete script", () => {
      if (confirm(`Delete the script ${script.name}? Its recordings stay in the project's sounds.`)) post("/api/script", { name: script.name, delete: true });
    });
    remove.className = "danger";
    head.append(remove);
  }
  const area = document.createElement("textarea");
  area.className = "script-text";
  area.spellcheck = true;
  area.value = script?.text ?? "";
  area.placeholder = "Write what is said, sentence by sentence.\n\nA line with --- starts a new part (recorded in one go).\nA line like [middle] names a section.";
  area.addEventListener("input", () => {
    status.textContent = "Not saved yet…";
    clearTimeout(timer);
    timer = setTimeout(() => save(area.value, name?.value.trim() ?? script.name, status), SAVE_DELAY);
  });
  const help = document.createElement("p");
  help.className = "hint";
  help.textContent =
    "Every word becomes a moment scenes can wait for or sync to: narration.square is the first \"square\", narration.square.2 the second, narration.middle.square the first after a [middle] line.";
  const parts = document.createElement("div");
  parts.className = "script-parts";
  elements.view.append(head, area, help, parts);
  if (script) drawParts(script);
}

async function save(text, name, status) {
  if (!/^[A-Za-z][\w-]*$/.test(name)) {
    status.textContent = "Give it a name: letters, digits, - and _, starting with a letter.";
    return;
  }
  status.textContent = "Saving…";
  if (await post("/api/script", { name, text })) {
    current = name;
    writingNew = false;
    status.textContent = "Saved";
    await refreshBin();
  } else {
    status.textContent = "Not saved";
  }
}

function drawParts(script) {
  const box = elements.view.querySelector(".script-parts");
  if (!box) return;
  box.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "Parts";
  box.append(heading);
  const { audio = [] } = page.state.timeline;
  script.parts.forEach((part, index) => {
    const row = document.createElement("div");
    row.className = "script-part";
    const title = document.createElement("span");
    title.className = "part-title";
    title.textContent = `${index + 1}. ${part.title || "(empty)"}`;
    const facts = document.createElement("span");
    facts.className = "hint";
    const placed = part.clip !== null ? audio[part.clip] : null;
    facts.textContent = [
      `${part.length.toFixed(1)} s`,
      part.take ? "recorded" : "not recorded yet",
      placed ? `on ${trackName("audio", placed.track)} at ${formatTime(placed.start)}` : "not on the timeline",
    ].join(" · ");
    row.append(title, facts);
    if (!placed) row.append(button("Place at the playhead", () => place(script.name, index, part.length)));
    box.append(row);
  });
}

/** Put a part on the first audio track with room for it at the playhead. */
function place(script, part, length) {
  const start = page.state.time;
  const audio = page.state.timeline.audio ?? [];
  let track = 0;
  while (audio.some((clip) => clip.track === track && clip.start < start + length && clip.end > start)) track += 1;
  post("/api/audio/add", { script, part, start, track });
}

function uniqueName(scripts) {
  const names = new Set(scripts.map((script) => script.name));
  let name = "narration";
  for (let number = 2; names.has(name); number++) name = `narration-${number}`;
  return name;
}

export function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds - minutes * 60).toFixed(1).padStart(4, "0")}`;
}
