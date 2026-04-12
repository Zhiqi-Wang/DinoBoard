import { getDifficultyPolicy } from "../../general-web/difficulty_controls.js";

export const OP_TARGET_LINES = [0, 1, 2, 3, 4, -1];
export const FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3];

export const state = {
  sessionId: null,
  humanPlayer: 0,
  sideMode: "first",
  difficultyMode: "experience",
  aiPolicy: getDifficultyPolicy("experience"),
  stateVersion: 0,
  current: null,
  legalActions: [],
  actionIndex: new Map(),
  actionIndexStateVersion: -1,
  actionsLoading: false,
  selectedPick: null, // { source, source_idx, color, sourceKey, el }
  forceMode: false,
  opsButtonsLocked: false,
};
