export const DIFFICULTY_CONFIG = {
  heuristic: { engine: "heuristic", simulations: 0, temperature: 0.0, label: "启发式" },
  experience: { engine: "netmcts", simulations: 10, temperature: 0.0, label: "体验" },
  expert: { engine: "netmcts", simulations: 500, temperature: 0.0, label: "专家" },
  master: { engine: "netmcts", simulations: 10000, temperature: 0.0, label: "大师" },
};

export function getDifficultyPolicy(mode) {
  return DIFFICULTY_CONFIG[mode] || DIFFICULTY_CONFIG.experience;
}

export function bindDifficultyButtons(config) {
  const { buttons, initialMode = "experience", onChange } = config;
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
      if (onChange) onChange(mode, getDifficultyPolicy(mode));
    });
  });

  render();
  if (onChange) onChange(mode, getDifficultyPolicy(mode));
}
