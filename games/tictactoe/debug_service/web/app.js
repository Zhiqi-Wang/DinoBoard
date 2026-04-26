import { api } from "../general-web/api_client.js";
import { createHintPanel } from "../general-web/hint_panel.js";
import { GAME_CATALOG, currentGameIdFromPath } from "../general-web/game_catalog.js";
import { createLegalActionsStore } from "../general-web/legal_actions_store.js";
import { setupTwoPlayerPageControls, tryAutoStart } from "../general-web/two_player_page_boot.js";
import { createTwoPlayerTurnRuntime, renderTopWithHint } from "../general-web/two_player_turn_runtime.js";
import {
  byId as $,
  setTextById as setMsg,
  mapDifficultyToPolicy,
  canHumanControlTurn,
  setButtonsDisabled,
  scoreFromPublicState,
} from "../general-web/two_player_turn_helpers.js";

const state = {
  sessionId: null,
  current: null,
  stateVersion: 0,
  humanPlayer: 0,
  sideMode: "first",
  difficultyMode: "experience",
  aiPolicy: { engine: "netmcts", simulations: 10, temperature: 0.0, time_budget_ms: 0, label: "体验" },
  forceMode: false,
  opsLocked: false,
};

function formatMoveText(moveInfo) {
  const actor = Number(moveInfo?.actor ?? 0);
  const actionId = Number(moveInfo?.action_id ?? -1);
  if (actionId < 0) return "玩家已完成一步";
  const row = Math.floor(actionId / 3) + 1;
  const col = (actionId % 3) + 1;
  return `玩家${actor} 落子到第${row}行第${col}列`;
}

const hintPanel = createHintPanel({
  turnEl: $("hint-turn-line"),
  messageEl: $("hint-message-line"),
  scoreEl: $("hint-score-line"),
  winrateEl: $("hint-winrate-line"),
  suggestionEl: $("hint-suggest-line"),
  introMessage: "玩法：在 3x3 棋盘上先连成一条线的一方获胜。",
  formatTurn: (currentPlayer) => `当前轮到：玩家${currentPlayer}（${currentPlayer === 0 ? "X" : "O"}）`,
  formatScore: (scoreInfo) => `当前分数：P0 ${scoreInfo.p0} / P1 ${scoreInfo.p1}`,
  formatOpponentMove: (moveInfo) => `对手动作：${formatMoveText(moveInfo)}`,
  formatSuggestedMove: (moveInfo) => `AI 提示：建议 ${formatMoveText(moveInfo)}`,
});

function mapPolicyForTicTacToe(mode, policyFromGeneral) {
  return mapDifficultyToPolicy(mode, policyFromGeneral, { experience: 80, expert: 800, master: 4000 });
}

function canHumanControlCurrentTurn() {
  return canHumanControlTurn(state.current, state.humanPlayer, state.forceMode);
}

function setOpsLocked(locked) {
  state.opsLocked = !!locked;
  setButtonsDisabled(["btn-undo", "btn-force"], state.opsLocked);
}

function computeScoreInfo(publicState) {
  return scoreFromPublicState(publicState);
}

async function refreshActions() {
  if (!state.sessionId) return [];
  return legalActionsStore.refreshActions();
}

async function playCell(cellId) {
  if (!state.sessionId || !state.current) return;
  if (!canHumanControlCurrentTurn()) return;
  const board = state.current.public_state.game.board;
  if (board[cellId] !== ".") return;
  const endpoint = state.forceMode ? "force-opponent-move" : "actions";
  const out = await api(`/api/v1/games/${state.sessionId}/${endpoint}`, "POST", {
    action_id: cellId,
    state_version: state.stateVersion,
  });
  state.current = out;
  state.stateVersion = out.state_version;
  if (state.opsLocked) {
    setOpsLocked(false);
  }
  if (state.forceMode) {
    state.forceMode = false;
    setMsg("ops-msg", "已完成一次替对手落子");
  }
  render();
  await turnRuntime.maybeAiRespondIfNeeded();
}

function renderBoard() {
  const boardEl = $("ttt-board");
  boardEl.innerHTML = "";
  const board = state.current?.public_state?.game?.board || [".", ".", ".", ".", ".", ".", ".", ".", "."];
  const canPlay = canHumanControlCurrentTurn();
  for (let i = 0; i < 9; i += 1) {
    const btn = document.createElement("button");
    btn.className = "ttt-cell";
    const v = board[i];
    if (v === "X") btn.classList.add("x");
    if (v === "O") btn.classList.add("o");
    btn.textContent = v === "." ? "" : v;
    btn.disabled = !canPlay || v !== ".";
    btn.addEventListener("click", () => playCell(i).catch((e) => setMsg("ops-msg", e.message)));
    boardEl.appendChild(btn);
  }
}

function render() {
  renderTopWithHint({
    state,
    hintPanel,
    byId: $,
    computeScoreInfo,
    drawTurnText: "对局结束：平局",
    drawMessageText: "本局已结束，结果为平局。",
  });
  renderBoard();
}

const turnRuntime = createTwoPlayerTurnRuntime({
  state,
  api,
  hintPanel,
  setMsg,
  setOpsLocked,
  canHumanControlCurrentTurn,
  isHumanTurn: canHumanControlCurrentTurn,
  render,
  refreshActions,
  refreshAfterAi: async () => turnRuntime.refreshState(),
  mapPolicy: mapPolicyForTicTacToe,
  gameLabelForStart: (humanPlayer, aiLabel, sessionId) =>
    `已开局 ${sessionId}，你执 ${humanPlayer === 0 ? "X" : "O"}，难度=${aiLabel}`,
});

const legalActionsStore = createLegalActionsStore({
  api,
  endpointBuilder: () => `/api/v1/games/${state.sessionId}/legal-actions`,
  getCurrentStateVersion: () => state.stateVersion,
  onApply: ({ actions }) => actions,
});

async function boot() {
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
      storageKey: "dino_tictactoe_debug_zoom_percent",
    },
    sidebarControls: {
      sideButtons: { first: $("btn-side-first"), second: $("btn-side-second"), random: $("btn-side-random") },
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
      onDifficultyChange: (mode, policyFromGeneral) => {
        turnRuntime.bindDifficulty(mode, policyFromGeneral);
      },
      onStart: () => turnRuntime.handleStart().catch((e) => setMsg("start-msg", e.message)),
      onUndo: () => turnRuntime.handleUndo().catch((e) => setMsg("ops-msg", e.message)),
      onForce: () => turnRuntime.handleForce().catch((e) => setMsg("ops-msg", e.message)),
      onHint: () => turnRuntime.handleHint().catch((e) => setMsg("ops-msg", e.message)),
    },
  });

  await tryAutoStart({
    start: () => turnRuntime.handleStart(),
    onError: (e) => setMsg("start-msg", e?.message || String(e)),
  });
}

boot();

