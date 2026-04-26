export function resolveHumanPlayer(sideMode) {
  if (sideMode === "first") return 0;
  if (sideMode === "second") return 1;
  return Math.random() < 0.5 ? 0 : 1;
}

export function bindSideModeButtons(config) {
  const { buttons, initialMode = "first", onChange } = config;
  let mode = initialMode;

  function render() {
    Object.entries(buttons).forEach(([key, el]) => {
      if (!el) return;
      el.classList.toggle("active", key === mode);
    });
  }

  Object.entries(buttons).forEach(([key, el]) => {
    if (!el) return;
    el.addEventListener("click", () => {
      mode = key;
      render();
      if (onChange) onChange(mode);
    });
  });

  render();
  if (onChange) onChange(mode);
}
