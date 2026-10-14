// Recording script parts, and the teleprompter that shows what to say.
//
// The teleprompter (the Teleprompter button, or T) shows over the frame
// what the script parts on the timeline say: the sentence that is due
// large, the one after it smaller, each appearing a moment before its
// first word is due (the setting voice.lead) so it can be read ahead; the
// word that is due lights up.
//
// Recording a part: select its clip on an audio track and press ● (or
// R). The microphone opens and records from then on; playback starts a
// few seconds before the part (voice.countdown) while 3, 2, 1 counts down
// over the frame, then the part's sentences show as they are due. Press ●
// again to stop (Escape throws the take away). The take is copied into
// the project, its start is where the part started (the countdown is
// recorded too, but cut off), and Whisper times its words (POST
// /api/script/take): scenes synced to them follow the real voice. One
// undo puts the plan back.
//
// Headphones keep the timeline's sound out of the microphone.

import { uploadSound, scriptPart } from "./bin.js";
import { say } from "./editing.js";

let page = null;
let prompting = false; // the teleprompter is on
let take = null; // while recording a part: { clip, recorder, stream, chunks, startedAt, partAt }

const elements = {
  prompter: document.getElementById("prompter"),
  toggle: document.getElementById("prompter-button"),
  countdown: document.getElementById("countdown"),
};

export function initRecording(shared) {
  page = shared;
  prompting = page.loadFlag("manimx.prompter");
  elements.toggle.setAttribute("aria-pressed", String(prompting));
  elements.toggle.addEventListener("click", () => setPrompting(!prompting));
  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input, select, textarea, dialog")) return;
    if (event.key.toLowerCase() === "t" && !event.ctrlKey && !event.metaKey && !event.altKey) setPrompting(!prompting);
    if (event.key === "Escape" && take) cancelTake();
  });
}

export const recordingPart = () => take !== null;

function setPrompting(on) {
  prompting = on;
  elements.toggle.setAttribute("aria-pressed", String(on));
  page.saveFlag("manimx.prompter", on);
  drawPrompter(page.state.time);
}

// -- the teleprompter -----------------------------------------------------------------

/** Every sentence of the placed script parts: its words' global times. */
function sentences() {
  const result = [];
  (page.state.timeline?.audio ?? []).forEach((clip) => {
    if (!clip.script) return;
    const part = scriptPart(clip.script, clip.part);
    if (!part) return;
    const cutIn = clip.cut?.[0] ?? 0;
    for (const word of part.words) {
      const at = clip.start + word.start - cutIn;
      if (at < clip.start - 1e-6 || at > clip.end + 1e-6) continue;
      const last = result[result.length - 1];
      if (last && last.clip === clip && last.sentence === word.sentence) last.words.push({ text: word.text, at });
      else result.push({ clip, sentence: word.sentence, words: [{ text: word.text, at }] });
    }
  });
  return result.sort((a, b) => a.words[0].at - b.words[0].at);
}

/** Called by app.js whenever the playhead moves. */
export function drawPrompter(time) {
  if (take) followTake(time);
  const on = prompting || take !== null;
  const lead = page.state.timeline?.voice?.lead ?? 1.5;
  const all = on ? sentences() : [];
  // The sentence shown: the last one whose lead time has come, while it is not long over.
  let index = -1;
  all.forEach((sentence, i) => {
    if (sentence.words[0].at - lead <= time) index = i;
  });
  const current = all[index];
  const over = current && time > current.words[current.words.length - 1].at + 2.5; // said, and a moment more
  elements.prompter.hidden = !current || over;
  if (!current || over) return;
  elements.prompter.replaceChildren(line(current, time, "now"));
  if (all[index + 1]) elements.prompter.append(line(all[index + 1], time, "next"));
}

function line(sentence, time, kind) {
  const element = document.createElement("p");
  element.className = `prompt ${kind}`;
  for (const word of sentence.words) {
    const span = document.createElement("span");
    span.textContent = `${word.text} `;
    if (kind === "now") span.className = word.at <= time ? "said" : "";
    element.append(span);
  }
  return element;
}

// -- recording a part -------------------------------------------------------------------

/** Record script part clip `index` (an audio clip of a script part). */
export async function recordPart(index) {
  const clip = page.state.timeline.audio[index];
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
  } catch (error) {
    say(`No microphone: ${error.message}`, true);
    return;
  }
  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => chunks.push(event.data));
  const started = new Promise((resolve) => recorder.addEventListener("start", resolve, { once: true }));
  recorder.start();
  await started;
  take = { clip, recorder, stream, chunks, startedAt: performance.now(), partAt: null };
  document.body.classList.add("recording-take");
  const countdown = page.state.timeline.voice?.countdown ?? 3;
  page.seek(Math.max(0, clip.start - countdown));
  say(`Recording part ${clip.part + 1} of ${clip.script}: speak when the countdown ends. ● stops, Escape throws it away.`);
  page.play();
}

/** While recording: the countdown, and when the part starts. */
function followTake(time) {
  const left = take.clip.start - time;
  elements.countdown.hidden = left <= 0 || !page.state.playing;
  if (left > 0) elements.countdown.textContent = String(Math.ceil(left));
  else if (take.partAt === null) take.partAt = performance.now(); // the part starts now: in the take, this far in
}

async function finishTake() {
  const { recorder, stream, chunks } = take;
  const stopped = new Promise((resolve) => recorder.addEventListener("stop", resolve, { once: true }));
  recorder.stop();
  page.pause();
  await stopped;
  for (const track of stream.getTracks()) track.stop();
  document.body.classList.remove("recording-take");
  elements.countdown.hidden = true;
  return new Blob(chunks, { type: recorder.mimeType });
}

/** Stop recording and use the take. */
export async function stopTake() {
  const { clip, startedAt, partAt, recorder } = take;
  const blob = await finishTake();
  take = null;
  if (partAt === null) {
    say("The part never started: nothing was kept.", true);
    return;
  }
  const suffix = recorder.mimeType.includes("ogg") ? "ogg" : recorder.mimeType.includes("mp4") ? "m4a" : "webm";
  const file = await uploadSound(blob, `${clip.script}-part-${clip.part + 1}.${suffix}`);
  if (!file) return;
  say("Listening to the take (Whisper)… the first time, it downloads its model.");
  const offset = Math.round(partAt - startedAt) / 1000;
  const answer = await fetch("/api/script/take", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ script: clip.script, part: clip.part, file, offset }),
  }).then((response) => response.json());
  if (answer.error) {
    say(answer.error, true);
    return;
  }
  page.setVersion(answer.picture);
  say(answer.problem ? `Recorded; ${answer.problem}` : `Recorded part ${clip.part + 1}: its words are timed to your voice. Ctrl+Z puts the plan back.`, Boolean(answer.problem));
}

async function cancelTake() {
  await finishTake();
  take = null;
  document.getElementById("record").setAttribute("aria-pressed", "false");
  say("Recording thrown away.");
}
