function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

export function initZoomControls(config) {
  const {
    targetEl,
    outBtnEl,
    inBtnEl,
    valueEl,
    minPercent = 60,
    maxPercent = 180,
    stepPercent = 10,
    initialPercent = 100,
    storageKey = "dino_zoom_percent",
  } = config;

  if (!targetEl || !outBtnEl || !inBtnEl || !valueEl) return null;

  const savedRaw = window.localStorage.getItem(storageKey);
  const saved = Number(savedRaw);
  let current = Number.isFinite(saved) ? saved : initialPercent;

  function apply(next) {
    current = clamp(Math.round(next), minPercent, maxPercent);
    targetEl.style.zoom = `${current}%`;
    valueEl.textContent = `${current}%`;
    outBtnEl.disabled = current <= minPercent;
    inBtnEl.disabled = current >= maxPercent;
    window.localStorage.setItem(storageKey, String(current));
  }

  outBtnEl.addEventListener("click", () => apply(current - stepPercent));
  inBtnEl.addEventListener("click", () => apply(current + stepPercent));
  apply(current);

  return {
    set(percent) {
      apply(percent);
    },
    get() {
      return current;
    },
  };
}
