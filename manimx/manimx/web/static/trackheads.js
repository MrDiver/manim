// Track headers: the left end of every track, as in a video editor.
//
// A track has a name (Video 1, Audio 2 until renamed: double-click it)
// and its switches: scene tracks hide (the eye: not drawn), mute (the
// speaker: its scenes' sounds) and lock; audio tracks mute (M), solo (S:
// only soloed tracks are heard), lock, and a volume slider. Locked tracks
// cannot be edited by mistake: their clips do not move or cut. Every
// change is an edit like any other (POST /api/track, undone with Ctrl+Z)
// and saved to manimx.json.

import { liveSender, post } from "./editing.js";

let page = null;

export function initTrackHeads(shared) {
  page = shared;
}

/** The settings of track `index` of `kind` ("scenes" or "audio"). */
export function track(kind, index) {
  return { name: "", hidden: false, locked: false, muted: false, solo: false, volume: 1, ...(page.state.timeline.tracks?.[kind]?.[index] ?? {}) };
}

export const isLocked = (kind, index) => track(kind, index).locked;

/** The name a track shows: its own, or Video n / Audio n. */
export function trackName(kind, index) {
  return track(kind, index).name || `${kind === "scenes" ? "Video" : "Audio"} ${index + 1}`;
}

/** Fill a track row's label with its name and switches. */
export function trackHead(row, kind, index) {
  const label = row.querySelector(".label");
  label.replaceChildren();
  label.classList.add("track-head");
  const settings = track(kind, index);
  const edit = (changes) => post("/api/track", { kind, index, ...changes });
  const name = document.createElement("span");
  name.className = "track-name";
  name.textContent = trackName(kind, index);
  name.title = "Double-click to rename";
  label.append(name);
  if (!page.state.timeline.editable) return;
  name.addEventListener("dblclick", () => {
    const input = document.createElement("input");
    input.className = "track-rename";
    input.value = settings.name || name.textContent;
    name.replaceWith(input);
    input.focus();
    input.select();
    const done = (save) => {
      if (save && input.value.trim() !== settings.name) edit({ name: input.value.trim() });
      else input.replaceWith(name);
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") done(true);
      if (event.key === "Escape") done(false);
    });
    input.addEventListener("blur", () => done(true));
  });
  const switches = document.createElement("span");
  switches.className = "track-switches";
  const toggle = (key, text, on, off) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `switch ${key}`;
    button.textContent = text;
    button.setAttribute("aria-pressed", String(settings[key]));
    button.title = settings[key] ? on : off;
    button.addEventListener("click", () => edit({ [key]: !settings[key] }));
    return button;
  };
  if (kind === "scenes") {
    switches.append(
      toggle("hidden", "👁", "Hidden: not drawn (click to show)", "Shown (click to hide this track)"),
      toggle("muted", "🔈", "Muted: its scenes' sounds are not heard (click to hear them)", "Its scenes' sounds are heard (click to mute)"),
    );
  } else {
    switches.append(
      toggle("muted", "M", "Muted (click to hear it)", "Mute this track"),
      toggle("solo", "S", "Solo: only soloed tracks are heard (click to stop)", "Solo: hear only this track (and other soloed ones)"),
    );
  }
  switches.append(toggle("locked", "🔒", "Locked: its clips cannot be moved or cut (click to unlock)", "Lock this track against changes"));
  label.append(switches);
  if (kind === "audio") label.append(volumeSlider(settings.volume, "/api/track", (volume) => ({ kind, index, volume })));
}

/** A volume slider, 0 to 200 % (the middle, 100 %, is the sound as it is),
 * sending `route` with `body(volume)` live while it is dragged (one undo step). */
export function volumeSlider(volume, route, body) {
  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "volume";
  slider.min = "0";
  slider.max = "2";
  slider.step = "0.01";
  slider.value = String(volume);
  const describe = () => (slider.title = `Volume ${Math.round(Number(slider.value) * 100)} % (double-click: 100 %)`);
  describe();
  let send = null;
  const sender = () => (send ??= liveSender(route, () => {}));
  slider.addEventListener("pointerdown", () => {
    send = liveSender(route, () => {});
    page.state.dragging = true; // the timeline is not redrawn under the slider meanwhile
  });
  slider.addEventListener("input", () => {
    describe();
    sender()(body(Number(slider.value)));
  });
  slider.addEventListener("change", async () => {
    await send?.idle();
    send = null; // released: the next drag is a step of its own
    page.state.dragging = false;
  });
  slider.addEventListener("dblclick", () => {
    slider.value = "1";
    describe();
    post(route, body(1));
  });
  return slider;
}
