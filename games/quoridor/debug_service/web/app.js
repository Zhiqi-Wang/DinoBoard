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
  legalActions: [],
  wallUsage: { p0: 0, p1: 0 },
  wallOwners: { h: new Map(), v: new Map() },
  humanPlayer: 0,
  sideMode: "first",
  difficultyMode: "experience",
  aiPolicy: { engine: "netmcts", simulations: 20, temperature: 0.0, time_budget_ms: 0, label: "体验" },
  forceMode: false,
  opsLocked: false,
};

const BOARD_SIZE = 9;

function formatMoveText(moveInfo) {
  const actionId = Number(moveInfo?.action_id ?? -1);
  const type = String(moveInfo?.type || "");
  const row = Number(moveInfo?.row ?? -1);
  const col = Number(moveInfo?.col ?? -1);
  if (actionId < 0) return "玩家已完成一步";
  if (type === "move") return `走子到 (${row + 1}, ${col + 1})`;
  if (type === "hwall") return `放横墙 @ (${row + 1}, ${col + 1})`;
  if (type === "vwall") return `放竖墙 @ (${row + 1}, ${col + 1})`;
  return `动作 #${actionId}`;
}

const hintPanel = createHintPanel({
  turnEl: $("hint-turn-line"),
  messageEl: $("hint-message-line"),
  scoreEl: $("hint-score-line"),
  winrateEl: $("hint-winrate-line"),
  suggestionEl: $("hint-suggest-line"),
  introMessage: "玩法：每回合走子或放墙；先到达对侧底线者胜。",
  formatTurn: (currentPlayer) => `当前轮到：玩家${currentPlayer}${currentPlayer === 0 ? "（A）" : "（B）"}`,
  formatScore: (scoreInfo) => `当前分数：P0 ${scoreInfo.p0} / P1 ${scoreInfo.p1}`,
  formatOpponentMove: (moveInfo) => `对手动作：${formatMoveText(moveInfo)}`,
  formatSuggestedMove: (moveInfo) => `AI 提示：${formatMoveText(moveInfo)}`,
});

function mapPolicyForQuoridor(mode, policyFromGeneral) {
  return mapDifficultyToPolicy(mode, policyFromGeneral, { experience: 120, expert: 800, master: 2400 });
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

function rcKey(r, c) {
  return `${r},${c}`;
}

function buildLegalMaps(actions) {
  const moves = new Map();
  const hwalls = new Map();
  const vwalls = new Map();
  for (const item of actions || []) {
    const actionId = Number(item?.action_id ?? -1);
    const type = String(item?.type || "");
    const row = Number(item?.row ?? -1);
    const col = Number(item?.col ?? -1);
    if (!Number.isFinite(actionId) || actionId < 0 || row < 0 || col < 0) continue;
    if (type === "move") {
      moves.set(rcKey(row, col), actionId);
    } else if (type === "hwall") {
      hwalls.set(rcKey(row, col), actionId);
    } else if (type === "vwall") {
      vwalls.set(rcKey(row, col), actionId);
    }
  }
  return { moves, hwalls, vwalls };
}

function buildWallAnchorSets(game) {
  const hAnchors = new Set();
  const vAnchors = new Set();
  for (const item of game?.horizontal_walls || []) {
    const r = Number(item?.row ?? -1);
    const c = Number(item?.col ?? -1);
    if (r < 0 || c < 0) continue;
    hAnchors.add(rcKey(r, c));
  }
  for (const item of game?.vertical_walls || []) {
    const r = Number(item?.row ?? -1);
    const c = Number(item?.col ?? -1);
    if (r < 0 || c < 0) continue;
    vAnchors.add(rcKey(r, c));
  }
  return { hAnchors, vAnchors };
}

function decodeWallAction(actionId) {
  const a = Number(actionId);
  if (!Number.isFinite(a) || a < 0) return null;
  if (a >= 81 && a < 145) {
    const x = a - 81;
    return { kind: "h", row: Math.floor(x / 8), col: x % 8 };
  }
  if (a >= 145 && a < 209) {
    const x = a - 145;
    return { kind: "v", row: Math.floor(x / 8), col: x % 8 };
  }
  return null;
}

async function submitBoardAction(actionId) {
  if (!state.sessionId || !state.current) return;
  if (!canHumanControlCurrentTurn()) return;
  const endpoint = state.forceMode ? "force-opponent-move" : "actions";
  const out = await api(`/api/v1/games/${state.sessionId}/${endpoint}`, "POST", {
    action_id: Number(actionId),
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
  setMsg("action-msg", "");
  await refreshWallUsage();
  render();
  await turnRuntime.maybeAiRespondIfNeeded();
}

function renderBoardInteractive() {
  const game = state.current?.public_state?.game;
  const boardEl = $("board-grid");
  const metaEl = $("board-meta");
  if (!boardEl) return;
  boardEl.innerHTML = "";
  if (!game) {
    if (metaEl) metaEl.textContent = "尚未开局";
    return;
  }

  const legal = buildLegalMaps(state.legalActions);
  const { hAnchors, vAnchors } = buildWallAnchorSets(game);
  const canPlay = canHumanControlCurrentTurn();
  const pawns = Array.isArray(game.pawns) ? game.pawns : [];
  const pawnByCell = new Map();
  for (const p of pawns) {
    const pr = Number(p?.row ?? -1);
    const pc = Number(p?.col ?? -1);
    const pid = Number(p?.player ?? -1);
    if (pr >= 0 && pc >= 0 && (pid === 0 || pid === 1)) {
      pawnByCell.set(rcKey(pr, pc), pid);
    }
  }

  for (let gr = 0; gr < 17; gr += 1) {
    for (let gc = 0; gc < 17; gc += 1) {
      if (gr % 2 === 0 && gc % 2 === 0) {
        const row = Math.floor(gr / 2);
        const col = Math.floor(gc / 2);
        const key = rcKey(row, col);
        const actionId = legal.moves.get(key);
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = "board-cell";
        if (canPlay && actionId != null) {
          cell.classList.add("legal");
          cell.addEventListener("click", () => {
            submitBoardAction(actionId).catch((e) => setMsg("action-msg", e.message));
          });
        }
        const pawn = pawnByCell.get(key);
        if (pawn != null) {
          const piece = document.createElement("div");
          piece.className = `pawn ${pawn === 0 ? "p0" : "p1"}`;
          piece.textContent = pawn === 0 ? "A" : "B";
          cell.appendChild(piece);
        }
        boardEl.appendChild(cell);
        continue;
      }

      if (gr % 2 === 1 && gc % 2 === 0) {
        const hr = Math.floor((gr - 1) / 2);
        const hc = Math.floor(gc / 2);
        const key = rcKey(hr, hc);
        const keyLeft = rcKey(hr, hc - 1);
        const el = document.createElement("button");
        el.type = "button";
        el.className = "edge-slot edge-h";
        if (hAnchors.has(key)) {
          el.classList.add("wall-anchor", "wall-on");
          const owner = state.wallOwners.h.get(key);
          if (owner === 0) el.classList.add("wall-p0");
          else if (owner === 1) el.classList.add("wall-p1");
        } else if (hAnchors.has(keyLeft)) {
          el.classList.add("wall-tail");
        } else if (hc <= BOARD_SIZE - 1) {
          const actionId = legal.hwalls.get(key) ?? legal.hwalls.get(keyLeft);
          if (canPlay && actionId != null) {
            el.classList.add("legal");
            el.addEventListener("click", () => {
              submitBoardAction(actionId).catch((e) => setMsg("action-msg", e.message));
            });
          } else {
            el.classList.add("edge-disabled");
          }
        } else {
          el.classList.add("edge-disabled");
        }
        boardEl.appendChild(el);
        continue;
      }

      if (gr % 2 === 0 && gc % 2 === 1) {
        const vr = Math.floor(gr / 2);
        const vc = Math.floor((gc - 1) / 2);
        const key = rcKey(vr, vc);
        const keyUp = rcKey(vr - 1, vc);
        const el = document.createElement("button");
        el.type = "button";
        el.className = "edge-slot edge-v";
        if (vAnchors.has(key)) {
          el.classList.add("wall-anchor", "wall-on");
          const owner = state.wallOwners.v.get(key);
          if (owner === 0) el.classList.add("wall-p0");
          else if (owner === 1) el.classList.add("wall-p1");
        } else if (vAnchors.has(keyUp)) {
          el.classList.add("wall-tail");
        } else if (vr <= BOARD_SIZE - 1) {
          const actionId = legal.vwalls.get(key) ?? legal.vwalls.get(keyUp);
          if (canPlay && actionId != null) {
            el.classList.add("legal");
            el.addEventListener("click", () => {
              submitBoardAction(actionId).catch((e) => setMsg("action-msg", e.message));
            });
          } else {
            el.classList.add("edge-disabled");
          }
        } else {
          el.classList.add("edge-disabled");
        }
        boardEl.appendChild(el);
        continue;
      }

      const cross = document.createElement("div");
      cross.className = "board-cross";
      boardEl.appendChild(cross);
    }
  }

  const wallsRemaining = Array.isArray(game.walls_remaining) ? game.walls_remaining : [0, 0];
  let crossingCount = 0;
  for (const k of hAnchors) {
    if (vAnchors.has(k)) crossingCount += 1;
  }
  if (metaEl) {
    const turnText = canPlay ? "可直接点击棋盘执行动作" : "等待对手行动";
    const totalPlaced = Number(state.wallUsage.p0 || 0) + Number(state.wallUsage.p1 || 0);
    const placedText = `已放墙：P0=${state.wallUsage.p0} / P1=${state.wallUsage.p1}`;
    const crossText = crossingCount > 0 ? `；警告：检测到同锚点交叉墙 ${crossingCount} 个` : "";
    metaEl.textContent =
      `墙剩余：P0=${Number(wallsRemaining[0] ?? 0)} / P1=${Number(wallsRemaining[1] ?? 0)}；${placedText}；总已放墙=${totalPlaced}；${turnText}${crossText}`;
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
  renderBoardInteractive();
}

async function refreshActions() {
  if (!state.sessionId) {
    state.legalActions = [];
    render();
    return [];
  }
  const actions = await legalActionsStore.refreshActions();
  state.legalActions = Array.isArray(actions) ? actions : [];
  render();
  return state.legalActions;
}

async function refreshWallUsage() {
  if (!state.sessionId) {
    state.wallUsage = { p0: 0, p1: 0 };
    state.wallOwners = { h: new Map(), v: new Map() };
    return;
  }
  try {
    const replay = await api(`/api/v1/games/${state.sessionId}/replay`);
    let p0 = 0;
    let p1 = 0;
    const hOwners = new Map();
    const vOwners = new Map();
    for (const ev of replay?.events || []) {
      const actionId = Number(ev?.action_id ?? -1);
      const actor = Number(ev?.actor ?? -1);
      if (actionId < 81) continue;
      if (actor === 0) p0 += 1;
      else if (actor === 1) p1 += 1;
      let wallInfo = null;
      const t = String(ev?.type || "");
      const r = Number(ev?.row ?? -1);
      const c = Number(ev?.col ?? -1);
      if ((t === "hwall" || t === "vwall") && r >= 0 && c >= 0) {
        wallInfo = { kind: t === "hwall" ? "h" : "v", row: r, col: c };
      } else {
        wallInfo = decodeWallAction(actionId);
      }
      if (wallInfo && (actor === 0 || actor === 1)) {
        const key = rcKey(wallInfo.row, wallInfo.col);
        if (wallInfo.kind === "h") hOwners.set(key, actor);
        else if (wallInfo.kind === "v") vOwners.set(key, actor);
      }
    }
    state.wallUsage = { p0, p1 };
    state.wallOwners = { h: hOwners, v: vOwners };
  } catch {
    // keep previous value on transient read failure
  }
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
  refreshAfterAi: async () => {
    await turnRuntime.refreshState();
    await refreshActions();
    await refreshWallUsage();
    render();
  },
  mapPolicy: mapPolicyForQuoridor,
  gameLabelForStart: (humanPlayer, aiLabel, sessionId) =>
    `已开局 ${sessionId}，你是玩家${humanPlayer}，难度=${aiLabel}`,
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
      storageKey: "dino_quoridor_debug_zoom_percent",
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
  await refreshWallUsage();
  render();
}

boot();

