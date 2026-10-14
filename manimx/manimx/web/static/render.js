// Rendering the project's video from the page (manimx/web/rendering.py).
//
// Render opens a dialog: the range (the whole timeline, or from the
// playhead), then Start. The server saves manimx.json and runs `manimx
// render` in a process of its own, at the project's full quality and with
// its H.264 encoder (the GPU's when one works), so editing goes on
// meanwhile. The dialog shows the progress (frames, frames per second,
// time left) while it runs, Cancel stops it, and the finished video plays
// in the dialog.
//
// "Sound for editing" renders the video without sound and, next to it, a
// WAV per track and a Reaper (.rpp) and OpenTimelineIO (.otio) session
// using them (manimx/drivers/session.py): open one in Reaper, Kdenlive or
// another program to mix and master the sound there.

const POLL = 400; // ms between progress requests

const dialog = document.getElementById("render-dialog");
const body = document.getElementById("render-body");
const openButton = document.getElementById("render-button");
let page = null;
let polling = null;
let stems = false; // "Sound for editing" is ticked

export function initRender(shared) {
  page = shared;
  openButton.addEventListener("click", async () => {
    dialog.showModal();
    await refresh();
  });
  dialog.addEventListener("close", () => {
    clearTimeout(polling);
    polling = null;
  });
}

/** Called when the timeline was loaded: the button only shows for a project. */
export function renderAvailable(editable) {
  openButton.hidden = !editable;
}

async function refresh() {
  clearTimeout(polling);
  let status;
  try {
    status = await (await fetch("/api/render")).json();
  } catch {
    status = { state: "failed", error: "the preview server does not answer" };
  }
  draw(status);
  if (["preparing", "rendering", "cancelling"].includes(status.state) && dialog.open) polling = setTimeout(refresh, POLL);
}

async function send(route, data) {
  const response = await fetch(route, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const answer = await response.json();
  if (!response.ok) draw({ state: "failed", error: answer.error });
  else refresh();
}

function draw(status) {
  body.replaceChildren();
  const running = ["preparing", "rendering", "cancelling"].includes(status.state);
  if (!running) body.append(rangeForm());
  if (status.state === "idle") return;
  const line = (text, className = "") => {
    const element = document.createElement("p");
    element.className = className;
    element.textContent = text;
    body.append(element);
    return element;
  };
  if (running) {
    const bar = document.createElement("progress");
    bar.max = status.total ?? 1;
    if (status.total) bar.value = status.done;
    body.append(bar);
    if (status.state === "preparing") line("Preparing the scenes…");
    else if (status.state === "cancelling") line("Cancelling…");
    else {
      const left = status.left == null ? "" : ` · ${Math.ceil(status.left)} s left`;
      line(`Frame ${status.done} of ${status.total} · ${status.fps ? status.fps.toFixed(1) : "–"} frames/s${left}`);
    }
    if (status.encoder) line(`Encoding with ${status.encoder}`, "hint");
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () => send("/api/render/cancel", {}));
    body.append(cancel);
    return;
  }
  if (status.state === "done") {
    const seconds = status.finished && status.started ? ` in ${(status.finished - status.started).toFixed(1)} s` : "";
    line(`Rendered ${status.total} frames${seconds} with ${status.encoder}: ${status.file}`);
    if (status.session) {
      line(`Sound for editing: ${status.stems} track${status.stems === 1 ? "" : "s"} as WAV files, and ${status.session} (Reaper) with an .otio beside it (Kdenlive, Resolve, …). The video has no sound of its own.`, "hint");
    }
    const video = document.createElement("video");
    video.controls = true;
    video.src = `/api/render/video?at=${status.finished}`;
    body.append(video);
  } else if (status.state === "cancelled") {
    line("Cancelled; the unfinished video was removed.", "hint");
  } else if (status.state === "failed") {
    line("The render failed:", "problem");
    const error = document.createElement("pre");
    error.textContent = status.error;
    body.append(error);
  }
}

/** Start and end fields and the Start button. */
function rangeForm() {
  const form = document.createElement("div");
  form.className = "render-range";
  const { duration } = page.state.timeline;
  const start = number(0, duration);
  const end = number(duration, duration);
  const fromPlayhead = document.createElement("button");
  fromPlayhead.type = "button";
  fromPlayhead.textContent = "From the playhead";
  fromPlayhead.addEventListener("click", () => {
    start.value = page.state.time.toFixed(3);
  });
  const go = document.createElement("button");
  go.type = "button";
  go.className = "primary";
  go.textContent = "Start";
  go.addEventListener("click", () => {
    const from = Number(start.value) || 0;
    const to = Number(end.value);
    send("/api/render", { start: from, end: Number.isFinite(to) && to < duration ? to : null, stems });
  });
  const separate = document.createElement("input");
  separate.type = "checkbox";
  separate.checked = stems;
  separate.addEventListener("change", () => {
    stems = separate.checked;
  });
  const choice = document.createElement("label");
  choice.className = "render-stems";
  choice.title = "Mix the sound in an audio program: the video comes without sound, each track's sound as a WAV file, and a session for Reaper (.rpp) and for Kdenlive or Resolve (.otio) with all of them in place";
  choice.append(separate, " Sound for editing (Reaper, Kdenlive): a WAV per track");
  form.append(label("From", start, "s"), label("To", end, "s"), fromPlayhead, choice, go);
  return form;
}

function number(value, maximum) {
  const input = document.createElement("input");
  input.type = "number";
  input.min = "0";
  input.max = String(maximum);
  input.step = "0.01";
  input.value = value.toFixed(3);
  return input;
}

function label(text, input, unit) {
  const element = document.createElement("label");
  element.append(`${text} `, input, ` ${unit}`);
  return element;
}
