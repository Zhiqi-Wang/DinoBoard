import { bindSideModeButtons } from "./opening_controls.js";
import { bindDifficultyButtons } from "./difficulty_controls.js";

export function bindSidebarControls(config) {
  const {
    sideButtons,
    difficultyButtons,
    startButton,
    undoButton,
    forceButton,
    hintButton,
    initialSideMode = "first",
    initialDifficultyMode = "experience",
    onSideModeChange,
    onDifficultyChange,
    onStart,
    onUndo,
    onForce,
    onHint,
  } = config;

  bindSideModeButtons({
    buttons: sideButtons,
    initialMode: initialSideMode,
    onChange: (mode) => {
      if (onSideModeChange) onSideModeChange(mode);
    },
  });

  bindDifficultyButtons({
    buttons: difficultyButtons,
    initialMode: initialDifficultyMode,
    onChange: (mode, policy) => {
      if (onDifficultyChange) onDifficultyChange(mode, policy);
    },
  });

  if (startButton) startButton.onclick = () => onStart && onStart();
  if (undoButton) undoButton.onclick = () => onUndo && onUndo();
  if (forceButton) forceButton.onclick = () => onForce && onForce();
  if (hintButton) hintButton.onclick = () => onHint && onHint();
}
