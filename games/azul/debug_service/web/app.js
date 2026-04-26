import { api } from "../general-web/api_client.js";
import { createHintPanel } from "../general-web/hint_panel.js";
import { GAME_CATALOG, currentGameIdFromPath } from "../general-web/game_catalog.js";
import { createLegalActionsStore } from "../general-web/legal_actions_store.js";
import { setupTwoPlayerPageControls, tryAutoStart } from "../general-web/two_player_page_boot.js";
import { createTwoPlayerTurnRuntime, renderTopWithHint } from "../general-web/two_player_turn_runtime.js";
import { canHumanControlTurn, scoreFromPublicState } from "../general-web/two_player_turn_helpers.js";

import { $, setMsg } from "./core/dom.js";
import { state, FLOOR_PENALTIES } from "./core/state.js";
import { tileClass, renderTileList, wallColorAt, pickKey, buildActionIndex } from "./core/utils.js";
import { renderSelectedPick } from "./render/selection_panel.js";
import { renderFactories, renderCenter } from "./render/factories_center.js";
import { renderPlayerBoard } from "./render/player_board.js";
import { createMoveFlow } from "./controllers/move_flow.js";

function clearPickSelection() {
  state.selectedPick = null;
  document.querySelectorAll(".tile-pick.selected").forEach((el) => el.classList.remove("selected"));
  renderSelectionPanel();
  rerenderBoardsFromCurrent();
}

function overwriteActionIndex(nextMap) {
  state.actionIndex.clear();
  for (const [k, v] of nextMap.entries()) {
    state.actionIndex.set(k, v);
  }
}

function syncOpsButtonsState() {
  const undoBtn = $("btn-undo");
  const forceBtn = $("btn-force");
  const disabled = !!state.opsButtonsLocked;
  if (undoBtn) undoBtn.disabled = disabled;
  if (forceBtn) forceBtn.disabled = disabled;
}

function setOpsButtonsLocked(locked) {
  state.opsButtonsLocked = !!locked;
  syncOpsButtonsState();
}

function formatMoveText(moveInfo) {
  const actor = Number(moveInfo?.actor ?? 0);
  const move = moveInfo?.move || {};
  const source = String(move?.source || "unknown");
  const sourceIdx = Number(move?.source_idx ?? -1);
  const color = String(move?.color || "?");
  const targetLine = Number(move?.target_line ?? -1);
  const targetText = targetLine < 0 ? "地板" : `第${targetLine}行`;
  const srcText = source === "center" ? "中心区" : `工厂${sourceIdx}`;
  return `玩家${actor} 从${srcText}拿${color}，落到${targetText}`;
}

const hintPanel = createHintPanel({
  turnEl: $("hint-turn-line"),
  messageEl: $("hint-message-line"),
  scoreEl: $("hint-score-line"),
  winrateEl: $("hint-winrate-line"),
  suggestionEl: $("hint-suggest-line"),
  introMessage: "玩法：先点击来源中的砖色，再点击己方板块目标行（或地板）。",
  formatTurn: (currentPlayer) => `当前轮到：玩家${currentPlayer}`,
  formatScore: (scoreInfo) => `当前分数：P0 ${scoreInfo.p0} / P1 ${scoreInfo.p1}`,
  formatOpponentMove: (moveInfo) => `对手动作：${formatMoveText(moveInfo)}`,
  formatSuggestedMove: (moveInfo) => `AI 提示：建议 ${formatMoveText(moveInfo)}`,
});

function computeScoreInfo(publicState) {
  return scoreFromPublicState(publicState);
}

function allowedTargetsForSelectedPick() {
  if (!state.selectedPick) return new Set();
  const targets = state.actionIndex.get(state.selectedPick.sourceKey);
  return new Set(targets ? [...targets.keys()] : []);
}

function rerenderBoardsFromCurrent() {
  if (!state.current) return;
  const game = state.current.public_state.game;
  const currentPlayer = state.current.public_state.common.current_player;
  const interactiveActor = (
    currentPlayer === state.humanPlayer ||
    (state.forceMode && currentPlayer !== state.humanPlayer)
  ) ? currentPlayer : null;
  const allowed = allowedTargetsForSelectedPick();
  renderPlayerBoard({
    rootEl: $("p0-board"),
    playerData: game.players[0],
    ownerPid: 0,
    currentPlayer,
    interactiveActor,
    selectedPick: state.selectedPick,
    allowedTargets: allowed,
    tileClass,
    renderTileList,
    wallColorAt,
    floorPenalties: FLOOR_PENALTIES,
    onTargetClick: (line) => moveFlow.playSelectedTarget(line).catch((e) => setMsg("ops-msg", e.message)),
  });
  renderPlayerBoard({
    rootEl: $("p1-board"),
    playerData: game.players[1],
    ownerPid: 1,
    currentPlayer,
    interactiveActor,
    selectedPick: state.selectedPick,
    allowedTargets: allowed,
    tileClass,
    renderTileList,
    wallColorAt,
    floorPenalties: FLOOR_PENALTIES,
    onTargetClick: (line) => moveFlow.playSelectedTarget(line).catch((e) => setMsg("ops-msg", e.message)),
  });
}

function canHumanControlCurrentTurn() {
  return canHumanControlTurn(state.current, state.humanPlayer, state.forceMode);
}

function renderSelectionPanel() {
  renderSelectedPick({
    selectedPick: state.selectedPick,
    selectedPickEl: $("selected-pick"),
    tileClass,
  });
}

function onPickSelected(pick) {
  document.querySelectorAll(".tile-pick.selected").forEach((el) => el.classList.remove("selected"));
  pick.el.classList.add("selected");
  state.selectedPick = pick;
  renderSelectionPanel();
  rerenderBoardsFromCurrent();
}

function renderState(payload) {
  state.current = payload;
  state.stateVersion = payload.state_version;
  if (payload.state_version === 0) {
    hintPanel.showIntro();
    hintPanel.setOpponentWinrate(null);
  }
  if (state.actionIndexStateVersion !== state.stateVersion) {
    state.actionIndex.clear();
    state.legalActions = [];
    state.actionIndexStateVersion = -1;
    state.selectedPick = null;
  }
  syncOpsButtonsState();
  const common = payload.public_state.common;
  if (state.forceMode && common.current_player === state.humanPlayer) {
    state.forceMode = false;
  }
  const game = payload.public_state.game;
  renderTopWithHint({
    state,
    hintPanel,
    byId: $,
    computeScoreInfo,
    drawTurnText: "对局结束：平局",
    drawMessageText: "本局已结束，结果为平局。",
  });
  renderFactories({
    factories: game.factories,
    rootEl: $("factories"),
    actionIndex: state.actionIndex,
    isActionIndexReady: () => !state.actionsLoading && state.actionIndexStateVersion === state.stateVersion,
    canPickSource: canHumanControlCurrentTurn,
    pickKey,
    tileClass,
    setOpsMsg: (msg) => setMsg("ops-msg", msg),
    onPick: onPickSelected,
  });
  renderCenter({
    tiles: game.center,
    firstPlayerTokenInCenter: game.first_player_token_in_center,
    rootEl: $("center"),
    actionIndex: state.actionIndex,
    isActionIndexReady: () => !state.actionsLoading && state.actionIndexStateVersion === state.stateVersion,
    canPickSource: canHumanControlCurrentTurn,
    pickKey,
    tileClass,
    setOpsMsg: (msg) => setMsg("ops-msg", msg),
    onPick: onPickSelected,
  });
  rerenderBoardsFromCurrent();
}

function renderActions(actions, actionStateVersion) {
  state.legalActions = actions;
  overwriteActionIndex(buildActionIndex(actions));
  state.actionIndexStateVersion = actionStateVersion;
  if (state.selectedPick && !state.actionIndex.has(state.selectedPick.sourceKey)) {
    clearPickSelection();
    return;
  }
  renderSelectionPanel();
  rerenderBoardsFromCurrent();
}

async function refreshState() {
  if (!state.sessionId) return;
  const st = await api(`/api/v1/games/${state.sessionId}/state`);
  renderState(st);
}

async function refreshActions() {
  if (!state.sessionId) return;
  return legalActionsStore.refreshActions();
}

async function refreshAll() {
  await refreshState();
  await refreshActions();
}

async function refreshAllFromPayload(statePayload) {
  renderState(statePayload);
  await refreshActions();
}

const turnRuntime = createTwoPlayerTurnRuntime({
  state,
  api,
  hintPanel,
  setMsg,
  setOpsLocked: setOpsButtonsLocked,
  canHumanControlCurrentTurn,
  isHumanTurn: canHumanControlCurrentTurn,
  render: () => {
    if (state.current) {
      renderState(state.current);
    }
  },
  refreshActions,
  refreshAfterAi: refreshAll,
  mapPolicy: (_mode, policy) => policy,
  gameLabelForStart: (_humanPlayer, aiLabel, sessionId) => `已创建 ${sessionId}（难度：${aiLabel}）`,
  onStartReset: clearPickSelection,
  onUndoReset: clearPickSelection,
  onForceReset: clearPickSelection,
  guardLimit: 32,
});

const moveFlow = createMoveFlow({
  state,
  getById: $,
  tileClass,
  api,
  refreshAllFromPayload,
  clearPickSelection,
  maybeAiRespondIfNeeded: () => turnRuntime.maybeAiRespondIfNeeded(),
  onPlayerOperated: () => setOpsButtonsLocked(false),
  setOpsMsg: (msg) => setMsg("ops-msg", msg),
});

const legalActionsStore = createLegalActionsStore({
  api,
  endpointBuilder: () => `/api/v1/games/${state.sessionId}/legal-actions`,
  getCurrentStateVersion: () => state.stateVersion,
  onLoadingChange: (loading) => {
    state.actionsLoading = !!loading;
  },
  onApply: ({ actions, data }) => {
    renderActions(actions || [], data?.state_version);
  },
});

async function boot() {
  window.addEventListener("resize", () => {
    const factories = state.current?.public_state?.game?.factories;
    if (factories) {
      renderFactories({
        factories,
        rootEl: $("factories"),
        actionIndex: state.actionIndex,
        isActionIndexReady: () => !state.actionsLoading && state.actionIndexStateVersion === state.stateVersion,
        canPickSource: canHumanControlCurrentTurn,
        pickKey,
        tileClass,
        setOpsMsg: (msg) => setMsg("ops-msg", msg),
        onPick: onPickSelected,
      });
    }
  });

  setupTwoPlayerPageControls({
    gameSelector: {
      selectEl: $("game-selector"),
      games: GAME_CATALOG,
      currentGameId: currentGameIdFromPath(window.location.pathname),
    },
    zoomControls: {
      targetEl: $("board-stage"),
      outBtnEl: $("btn-zoom-out"),
      inBtnEl: $("btn-zoom-in"),
      valueEl: $("zoom-value"),
      minPercent: 60,
      maxPercent: 180,
      stepPercent: 10,
      initialPercent: 100,
      storageKey: "dino_azul_debug_zoom_percent",
    },
    sidebarControls: {
      sideButtons: {
        first: $("btn-side-first"),
        second: $("btn-side-second"),
        random: $("btn-side-random"),
      },
      difficultyButtons: {
        heuristic: $("btn-diff-heuristic"),
        experience: $("btn-diff-experience"),
        expert: $("btn-diff-expert"),
        master: $("btn-diff-master"),
      },
      startButton: $("btn-start"),
      undoButton: $("btn-undo"),
      forceButton: $("btn-force"),
      hintButton: $("btn-hint"),
      initialSideMode: "first",
      initialDifficultyMode: "experience",
      onSideModeChange: (mode) => {
        state.sideMode = mode;
      },
      onDifficultyChange: (mode, policy) => {
        state.difficultyMode = mode;
        state.aiPolicy = policy;
      },
      onStart: () => turnRuntime.handleStart().catch((e) => setMsg("start-msg", e.message)),
      onUndo: () => turnRuntime.handleUndo().catch((e) => setMsg("ops-msg", e.message)),
      onForce: () => turnRuntime.handleForce().catch((e) => setMsg("ops-msg", e.message)),
      onHint: () => turnRuntime.handleHint().catch((e) => setMsg("ops-msg", e.message)),
    },
  });
  setOpsButtonsLocked(false);
  renderSelectionPanel();
  await tryAutoStart({
    start: () => turnRuntime.handleStart(),
    onError: (e) => setMsg("start-msg", e?.message || String(e)),
  });
}

boot();
