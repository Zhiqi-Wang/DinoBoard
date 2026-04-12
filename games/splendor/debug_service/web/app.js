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

const COLOR_NAMES = ["white", "blue", "green", "red", "black", "gold"];
const COLOR_LABELS = ["白", "蓝", "绿", "红", "黑", "金"];
const TAKE_THREE_COMBOS = [
  [0, 1, 2],
  [0, 1, 3],
  [0, 1, 4],
  [0, 2, 3],
  [0, 2, 4],
  [0, 3, 4],
  [1, 2, 3],
  [1, 2, 4],
  [1, 3, 4],
  [2, 3, 4],
];
const TAKE_TWO_DIFFERENT_COMBOS = [
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [1, 2],
  [1, 3],
  [1, 4],
  [2, 3],
  [2, 4],
  [3, 4],
];
const BUY_FACEUP_OFFSET = 0;
const RESERVE_FACEUP_OFFSET = 12;
const RESERVE_DECK_OFFSET = 24;
const BUY_RESERVED_OFFSET = 27;
const TAKE_THREE_OFFSET = 30;
const TAKE_TWO_DIFFERENT_OFFSET = 40;
const TAKE_ONE_OFFSET = 50;
const TAKE_TWO_SAME_OFFSET = 55;
const CHOOSE_NOBLE_OFFSET = 60;
const RETURN_TOKEN_OFFSET = 63;
const PASS_ACTION = 69;

const state = {
  sessionId: null,
  current: null,
  stateVersion: 0,
  legalActions: [],
  legalSet: new Set(),
  humanPlayer: 0,
  sideMode: "first",
  difficultyMode: "experience",
  aiPolicy: { engine: "netmcts", simulations: 10, temperature: 0.0, time_budget_ms: 0, label: "体验" },
  forceMode: false,
  opsLocked: false,
  gemPick: { colors: [] },
  tableauView: {
    sessionId: null,
    lastStateVersion: -1,
    tiers: [
      [null, null, null, null],
      [null, null, null, null],
      [null, null, null, null],
    ],
  },
};

function getGameState(payload) {
  return payload?.public_state?.game || null;
}

function getPlayerState(payload, playerIdx) {
  return getGameState(payload)?.players?.[playerIdx] || null;
}

function listTableauCardIds(payload) {
  const tableau = getGameState(payload)?.tableau || [];
  const ids = [];
  for (const tier of tableau) {
    for (const c of tier || []) {
      const cid = Number(c?.id ?? -1);
      if (cid >= 0) ids.push(cid);
    }
  }
  return ids;
}

function getCardId(card) {
  const id = Number(card?.id ?? -1);
  return Number.isFinite(id) ? id : -1;
}

function toFixedFourCards(rawTier) {
  const out = [null, null, null, null];
  let write = 0;
  for (const card of Array.isArray(rawTier) ? rawTier : []) {
    if (write >= 4) break;
    if (getCardId(card) < 0) continue;
    out[write] = card;
    write += 1;
  }
  return out;
}

function resetTableauViewState() {
  state.tableauView = {
    sessionId: state.sessionId,
    lastStateVersion: Number(state.stateVersion ?? -1),
    tiers: [
      [null, null, null, null],
      [null, null, null, null],
      [null, null, null, null],
    ],
  };
}

function buildTableauRenderModel() {
  const gameState = state.current?.public_state?.game || {};
  const tableauRaw = Array.isArray(gameState.tableau) ? gameState.tableau : [[], [], []];
  const stateVersion = Number(state.current?.state_version ?? state.stateVersion ?? -1);
  const shouldResetLayout =
    state.tableauView.sessionId !== state.sessionId ||
    stateVersion <= Number(state.tableauView.lastStateVersion ?? -1);

  if (shouldResetLayout) {
    state.tableauView.sessionId = state.sessionId;
    state.tableauView.lastStateVersion = stateVersion;
    state.tableauView.tiers = [toFixedFourCards(tableauRaw[0]), toFixedFourCards(tableauRaw[1]), toFixedFourCards(tableauRaw[2])];
  } else {
    for (let tier = 0; tier < 3; tier += 1) {
      const prevSlots = Array.isArray(state.tableauView.tiers[tier])
        ? [...state.tableauView.tiers[tier]]
        : [null, null, null, null];
      while (prevSlots.length < 4) prevSlots.push(null);
      if (prevSlots.length > 4) prevSlots.length = 4;

      const incomingCards = [];
      const incomingById = new Map();
      for (const card of Array.isArray(tableauRaw[tier]) ? tableauRaw[tier] : []) {
        const cid = getCardId(card);
        if (cid < 0 || incomingById.has(cid)) continue;
        incomingCards.push(card);
        incomingById.set(cid, card);
      }

      const nextSlots = [null, null, null, null];
      const consumedIds = new Set();

      // Keep cards that still exist at their previous visual slots.
      for (let slot = 0; slot < 4; slot += 1) {
        const prevCard = prevSlots[slot];
        const cid = getCardId(prevCard);
        if (cid >= 0 && incomingById.has(cid)) {
          nextSlots[slot] = incomingById.get(cid);
          consumedIds.add(cid);
        }
      }

      // Fill empty visual slots with newly arrived cards.
      const newcomers = incomingCards.filter((card) => !consumedIds.has(getCardId(card)));
      let newcomerIdx = 0;
      for (let slot = 0; slot < 4; slot += 1) {
        if (nextSlots[slot] != null) continue;
        if (newcomerIdx >= newcomers.length) break;
        nextSlots[slot] = newcomers[newcomerIdx];
        newcomerIdx += 1;
      }

      state.tableauView.tiers[tier] = nextSlots;
    }
    state.tableauView.lastStateVersion = stateVersion;
  }

  const logicalSlotMaps = [new Map(), new Map(), new Map()];
  for (let tier = 0; tier < 3; tier += 1) {
    const rawTier = Array.isArray(tableauRaw[tier]) ? tableauRaw[tier] : [];
    for (let slot = 0; slot < rawTier.length; slot += 1) {
      const cid = getCardId(rawTier[slot]);
      if (cid >= 0 && slot < 4) {
        logicalSlotMaps[tier].set(cid, slot);
      }
    }
  }

  return {
    visualTiers: state.tableauView.tiers,
    logicalSlotMaps,
    hasTableauPayload: Array.isArray(gameState.tableau) && gameState.tableau.length > 0,
  };
}

function listReservedCardIds(payload, playerIdx) {
  const reserved = getPlayerState(payload, playerIdx)?.reserved || [];
  const ids = [];
  for (const item of reserved) {
    const cid = Number(item?.card?.id ?? -1);
    if (cid >= 0) ids.push(cid);
  }
  return ids;
}

function countDiff(a = [], b = []) {
  const m = new Map();
  for (const v of a) m.set(v, (m.get(v) || 0) + 1);
  for (const v of b) m.set(v, (m.get(v) || 0) - 1);
  return m;
}

function buildTransitionPlan(prevPayload, nextPayload) {
  if (!prevPayload || !nextPayload) return null;
  const prevCommon = prevPayload?.public_state?.common;
  const nextCommon = nextPayload?.public_state?.common;
  if (!prevCommon || !nextCommon) return null;
  const prevVersion = Number(prevPayload?.state_version ?? -1);
  const nextVersion = Number(nextPayload?.state_version ?? -1);
  if (!(nextVersion > prevVersion)) return null;
  const actor = Number(prevCommon.current_player ?? 0);
  const prevPlayer = getPlayerState(prevPayload, actor);
  const nextPlayer = getPlayerState(nextPayload, actor);
  const prevGame = getGameState(prevPayload);
  const nextGame = getGameState(nextPayload);
  if (!prevPlayer || !nextPlayer || !prevGame || !nextGame) return null;

  const gemTransfers = [];
  const prevGems = Array.isArray(prevPlayer.gems) ? prevPlayer.gems : [];
  const nextGems = Array.isArray(nextPlayer.gems) ? nextPlayer.gems : [];
  const prevBank = Array.isArray(prevGame.bank) ? prevGame.bank : [];
  const nextBank = Array.isArray(nextGame.bank) ? nextGame.bank : [];
  for (let c = 0; c < 6; c += 1) {
    const pDelta = Number(nextGems[c] ?? 0) - Number(prevGems[c] ?? 0);
    const bDelta = Number(nextBank[c] ?? 0) - Number(prevBank[c] ?? 0);
    if (pDelta > 0 && bDelta < 0) {
      const count = Math.min(pDelta, -bDelta);
      for (let i = 0; i < count; i += 1) gemTransfers.push({ type: "bank_to_player", color: c, actor });
    } else if (pDelta < 0 && bDelta > 0) {
      const count = Math.min(-pDelta, bDelta);
      for (let i = 0; i < count; i += 1) gemTransfers.push({ type: "player_to_bank", color: c, actor });
    }
  }

  let buy = null;
  const cardsDelta = Number(nextPlayer.cards_count ?? 0) - Number(prevPlayer.cards_count ?? 0);
  if (cardsDelta > 0) {
    const prevBonuses = Array.isArray(prevPlayer.bonuses) ? prevPlayer.bonuses : [];
    const nextBonuses = Array.isArray(nextPlayer.bonuses) ? nextPlayer.bonuses : [];
    let bonusColor = -1;
    for (let c = 0; c < 5; c += 1) {
      if (Number(nextBonuses[c] ?? 0) > Number(prevBonuses[c] ?? 0)) {
        bonusColor = c;
        break;
      }
    }
    const reservedDiff = countDiff(
      listReservedCardIds(prevPayload, actor),
      listReservedCardIds(nextPayload, actor)
    );
    const removedReserved = [...reservedDiff.entries()].filter(([, d]) => d > 0).map(([id]) => id);
    const tableauDiff = countDiff(listTableauCardIds(prevPayload), listTableauCardIds(nextPayload));
    const removedTableau = [...tableauDiff.entries()].filter(([, d]) => d > 0).map(([id]) => id);
    const sourceId = removedReserved[0] ?? removedTableau[0] ?? -1;
    const sourceZone = removedReserved.length > 0 ? "reserved" : "tableau";
    if (bonusColor >= 0) {
      buy = { actor, bonusColor, sourceId, sourceZone };
    }
  }

  if (gemTransfers.length === 0 && !buy) return null;
  return { actor, gemTransfers, buy };
}

function capturePlanRects(plan) {
  const out = {};
  if (!plan?.buy || Number(plan.buy.sourceId) < 0) return out;
  const cid = String(plan.buy.sourceId);
  let cardEl = null;
  if (plan.buy.sourceZone === "reserved") {
    cardEl = document.querySelector(`#p${plan.buy.actor}-reserved .dev-card[data-card-id="${cid}"]`);
  }
  if (!cardEl) {
    cardEl = document.querySelector(`#tableau-root .dev-card[data-card-id="${cid}"]`);
  }
  if (cardEl) {
    out.buySourceRect = cardEl.getBoundingClientRect();
  }
  return out;
}

function animateChipFlight(fromRect, toRect, colorIdx, delayMs = 0) {
  if (!fromRect || !toRect) return;
  const token = document.createElement("div");
  token.className = `fx-fly-chip ${COLOR_NAMES[Math.max(0, Math.min(5, colorIdx))]}`;
  const startX = fromRect.left + fromRect.width / 2 - 16;
  const startY = fromRect.top + fromRect.height / 2 - 16;
  const endX = toRect.left + toRect.width / 2 - 16;
  const endY = toRect.top + toRect.height / 2 - 16;
  token.style.left = `${startX}px`;
  token.style.top = `${startY}px`;
  token.style.transform = "translate(0, 0) scale(1)";
  token.style.opacity = "1";
  document.body.appendChild(token);
  const dx = endX - startX;
  const dy = endY - startY;
  window.setTimeout(() => {
    token.style.transform = `translate(${dx}px, ${dy}px) scale(0.86)`;
    token.style.opacity = "0.18";
  }, delayMs + 32);
  window.setTimeout(() => token.remove(), delayMs + 2080);
}

function playTransitionAnimations(plan, preRects = {}) {
  if (!plan) return;
  let delay = 0;
  for (const tx of plan.gemTransfers || []) {
    const color = tx.color;
    const bankEl = document.querySelector(`#bank-row .token-chip.${COLOR_NAMES[color]}`);
    const playerEl = document.querySelector(`#p${tx.actor}-gems .token-chip.${COLOR_NAMES[color]}`);
    if (!bankEl || !playerEl) continue;
    const fromRect = tx.type === "bank_to_player" ? bankEl.getBoundingClientRect() : playerEl.getBoundingClientRect();
    const toRect = tx.type === "bank_to_player" ? playerEl.getBoundingClientRect() : bankEl.getBoundingClientRect();
    animateChipFlight(fromRect, toRect, color, delay);
    delay += 280;
  }

  if (plan.buy) {
    const bonusTarget = document.querySelector(`#p${plan.buy.actor}-bonuses .token-chip.${COLOR_NAMES[plan.buy.bonusColor]}`);
    const fromRect = preRects.buySourceRect;
    if (fromRect && bonusTarget) {
      animateChipFlight(fromRect, bonusTarget.getBoundingClientRect(), plan.buy.bonusColor, delay + 320);
    }
  }
}

function decodeAction(actionId) {
  const a = Number(actionId);
  if (a >= BUY_FACEUP_OFFSET && a < RESERVE_FACEUP_OFFSET) {
    const idx = a - BUY_FACEUP_OFFSET;
    return { type: "buy_faceup", tier: Math.floor(idx / 4), slot: idx % 4, label: `购买 T${Math.floor(idx / 4) + 1} #${(idx % 4) + 1}` };
  }
  if (a >= RESERVE_FACEUP_OFFSET && a < RESERVE_DECK_OFFSET) {
    const idx = a - RESERVE_FACEUP_OFFSET;
    return {
      type: "reserve_faceup",
      tier: Math.floor(idx / 4),
      slot: idx % 4,
      label: `保留明牌 T${Math.floor(idx / 4) + 1} #${(idx % 4) + 1}`,
    };
  }
  if (a >= RESERVE_DECK_OFFSET && a < BUY_RESERVED_OFFSET) {
    const tier = a - RESERVE_DECK_OFFSET;
    return { type: "reserve_deck", tier, label: `保留牌堆 T${tier + 1}` };
  }
  if (a >= BUY_RESERVED_OFFSET && a < TAKE_THREE_OFFSET) {
    const slot = a - BUY_RESERVED_OFFSET;
    return { type: "buy_reserved", slot, label: `购买保留牌 #${slot + 1}` };
  }
  if (a >= TAKE_THREE_OFFSET && a < TAKE_TWO_DIFFERENT_OFFSET) {
    const comb = TAKE_THREE_COMBOS[a - TAKE_THREE_OFFSET];
    const names = comb.map((c) => COLOR_NAMES[c]).join(" + ");
    return { type: "take_three", colors: comb, label: `拿三色：${names}` };
  }
  if (a >= TAKE_TWO_DIFFERENT_OFFSET && a < TAKE_ONE_OFFSET) {
    const comb = TAKE_TWO_DIFFERENT_COMBOS[a - TAKE_TWO_DIFFERENT_OFFSET];
    const names = comb.map((c) => COLOR_NAMES[c]).join(" + ");
    return { type: "take_two_different", colors: comb, label: `拿两色：${names}` };
  }
  if (a >= TAKE_ONE_OFFSET && a < TAKE_TWO_SAME_OFFSET) {
    const c = a - TAKE_ONE_OFFSET;
    return { type: "take_one", color: c, label: `拿一枚：${COLOR_NAMES[c]}` };
  }
  if (a >= TAKE_TWO_SAME_OFFSET && a < CHOOSE_NOBLE_OFFSET) {
    const c = a - TAKE_TWO_SAME_OFFSET;
    return { type: "take_two_same", color: c, label: `拿两枚同色：${COLOR_NAMES[c]}` };
  }
  if (a >= CHOOSE_NOBLE_OFFSET && a < RETURN_TOKEN_OFFSET) {
    const slot = a - CHOOSE_NOBLE_OFFSET;
    return { type: "choose_noble", slot, label: `选择贵族 #${slot + 1}` };
  }
  if (a >= RETURN_TOKEN_OFFSET && a < PASS_ACTION) {
    const c = a - RETURN_TOKEN_OFFSET;
    return { type: "return_token", color: c, label: `返还：${COLOR_NAMES[c]}` };
  }
  if (a === PASS_ACTION) return { type: "pass", label: "Pass" };
  return { type: "unknown", label: `动作 ${a}` };
}

function resolveBuyBonusLabel(moveInfo, payload) {
  const action = decodeAction(moveInfo?.action_id ?? -1);
  const directBonus = Number(moveInfo?.bought_bonus ?? -1);
  if (Number.isFinite(directBonus) && directBonus >= 0 && directBonus < COLOR_LABELS.length) {
    return `（提供${COLOR_LABELS[directBonus]}宝石）`;
  }
  if (action.type !== "buy_faceup") return "";
  const tableau = payload?.public_state?.game?.tableau;
  if (!Array.isArray(tableau)) return "";
  const tierCards = tableau[action.tier];
  if (!Array.isArray(tierCards)) return "";
  const card = tierCards[action.slot];
  const bonus = Number(card?.bonus ?? -1);
  if (!Number.isFinite(bonus) || bonus < 0 || bonus >= COLOR_LABELS.length) return "";
  return `（提供${COLOR_LABELS[bonus]}宝石）`;
}

function formatMoveText(moveInfo, payload = state.current) {
  const action = decodeAction(moveInfo?.action_id ?? -1);
  const actor = Number(moveInfo?.actor ?? 0);
  const bonusText = resolveBuyBonusLabel(moveInfo, payload);
  const tailText = moveInfo?.tail_solved ? "（残局已解析）" : "";
  return `玩家${actor}：${action.label}${bonusText}${tailText}`;
}

const hintPanel = createHintPanel({
  turnEl: $("hint-turn-line"),
  messageEl: $("hint-message-line"),
  scoreEl: $("hint-score-line"),
  winrateEl: $("hint-winrate-line"),
  suggestionEl: $("hint-suggest-line"),
  introMessage: "玩法：拿宝石、买牌、抢贵族；先触发 15 分终局者有机会获胜。",
  formatTurn: (currentPlayer) => `当前轮到：玩家${currentPlayer}`,
  formatScore: (scoreInfo) => `当前分数：P0 ${scoreInfo.p0} / P1 ${scoreInfo.p1}`,
  formatOpponentMove: (moveInfo) => `对手动作：${formatMoveText(moveInfo)}`,
  formatSuggestedMove: (moveInfo) => `AI 提示：${formatMoveText(moveInfo)}`,
});

function mapPolicy(mode, policyFromGeneral) {
  const base = mapDifficultyToPolicy(mode, policyFromGeneral, { expert: 5000, master: 35000 });
  if (base.engine !== "netmcts") return base;
  return {
    ...base,
    search_options: {
      enable_tail_solve: true,
      tail_solve_start_ply: 40,
      tail_solve_node_budget: 1000000,
      tail_solve_time_ms: 0,
      tail_solve_depth_limit: 5,
      tail_solve_score_diff_weight: 0.01,
    },
  };
}

function canHumanControlCurrentTurn() {
  return canHumanControlTurn(state.current, state.humanPlayer, state.forceMode);
}

function setOpsLocked(locked) {
  state.opsLocked = !!locked;
  setButtonsDisabled(["btn-undo", "btn-force"], state.opsLocked);
}

function getScoreInfo(publicState) {
  return scoreFromPublicState(publicState);
}

function tokenChip(colorIdx, count, options = {}) {
  const { onClick = null, selected = false, disabled = false } = options;
  const el = document.createElement("button");
  el.className = `token-chip ${COLOR_NAMES[colorIdx]}`;
  if (selected) el.classList.add("selected");
  el.type = "button";
  el.textContent = `${COLOR_LABELS[colorIdx]}:${count}`;
  const clickable = typeof onClick === "function" && !disabled;
  if (clickable) el.classList.add("clickable");
  el.disabled = !clickable;
  if (clickable) {
    el.addEventListener("click", () => onClick());
  }
  return el;
}

function renderCard(card, options = {}) {
  const root = document.createElement("div");
  root.className = "dev-card";
  const cardId = Number(card?.id ?? -1);
  if (cardId >= 0) root.dataset.cardId = String(cardId);
  const tierText = `T${Number(card?.tier ?? 0)}`;
  const pointsText = `★${Number(card?.points ?? 0)}`;
  const bonus = Number(card?.bonus ?? 0);
  const bonusLabel = COLOR_LABELS[Math.max(0, Math.min(4, bonus))];

  const head = document.createElement("div");
  head.className = "dev-card-head";
  const left = document.createElement("span");
  left.className = "dev-card-head-left";
  const tierEl = document.createElement("span");
  tierEl.textContent = tierText;
  const bonusEl = document.createElement("span");
  bonusEl.className = `bonus-inline ${COLOR_NAMES[Math.max(0, Math.min(4, bonus))]}`;
  bonusEl.textContent = bonusLabel;
  left.append(tierEl, bonusEl);
  const right = document.createElement("span");
  right.textContent = pointsText;
  head.append(left, right);
  root.appendChild(head);

  const costRow = document.createElement("div");
  costRow.className = "cost-row";
  const cost = Array.isArray(card?.cost) ? card.cost : [];
  for (let i = 0; i < 5; i += 1) {
    const v = Number(cost[i] ?? 0);
    if (v <= 0) continue;
    const pill = document.createElement("span");
    pill.className = `cost-gem ${COLOR_NAMES[i]}`;
    pill.textContent = String(v);
    costRow.appendChild(pill);
  }
  root.appendChild(costRow);

  const actions = Array.isArray(options.actions) ? options.actions : [];
  if (actions.length > 0) {
    const hasEnabledAction = actions.some(
      (action) => !action?.disabled && typeof action?.onClick === "function"
    );
    if (hasEnabledAction) {
      root.classList.add("clickable");
    }
    const actionRow = document.createElement("div");
    actionRow.className = "card-op-row";
    for (const action of actions) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "card-op-btn";
      btn.textContent = String(action.label || "操作");
      btn.disabled = !!action.disabled;
      if (!btn.disabled && typeof action.onClick === "function") {
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          action.onClick();
        });
      }
      actionRow.appendChild(btn);
    }
    root.appendChild(actionRow);
  }
  return root;
}

function renderReserved(ownerIdx, targetElId) {
  const container = $(targetElId);
  container.innerHTML = "";
  const players = state.current?.public_state?.game?.players || [];
  const reserved = players?.[ownerIdx]?.reserved || [];
  for (let slot = 0; slot < reserved.length; slot += 1) {
    const item = reserved[slot];
    const visibleToOpponent = !!item.visible_to_opponent;
    const isSelf = ownerIdx === state.humanPlayer;
    if (!isSelf && !visibleToOpponent) {
      const hiddenTier = Number(item?.card?.tier ?? 0);
      const tierLabel = Number.isFinite(hiddenTier) && hiddenTier > 0 ? `T${hiddenTier}` : "T?";
      const hidden = document.createElement("div");
      hidden.className = "reserve-hidden";
      hidden.textContent = `暗保留 ${tierLabel}`;
      container.appendChild(hidden);
      continue;
    }
    const buyActionId = 27 + slot;
    const canBuyReserved = isSelf && canHumanControlCurrentTurn() && state.legalSet.has(buyActionId);
    const actions = canBuyReserved
      ? [{ label: "购买保留", onClick: () => playAction(buyActionId).catch((e) => setMsg("ops-msg", e.message)) }]
      : [];
    container.appendChild(renderCard(item.card || {}, { actions }));
  }
}

function renderPlayers() {
  const players = state.current?.public_state?.game?.players || [];
  const currentPlayer = Number(state.current?.public_state?.common?.current_player ?? 0);
  for (let p = 0; p < 2; p += 1) {
    const pd = players[p] || {};
    const title = `玩家${p}${p === state.humanPlayer ? "（你）" : ""}${p === currentPlayer ? " · 当前" : ""}`;
    const metaText = `分:${Number(pd.points ?? 0)}  卡:${Number(pd.cards_count ?? 0)}  贵族:${Number(pd.nobles_count ?? 0)}`;
    const titleEl = $(`p${p}-title`);
    titleEl.innerHTML = "";
    const titleMain = document.createElement("span");
    titleMain.textContent = title;
    const titleMeta = document.createElement("span");
    titleMeta.className = "player-title-meta";
    titleMeta.textContent = metaText;
    titleEl.append(titleMain, titleMeta);
    $(`p${p}-meta`).textContent = "";
    const gems = $(`p${p}-gems`);
    gems.innerHTML = "";
    const gv = Array.isArray(pd.gems) ? pd.gems : [];
    const gemsTotal = gv.reduce((sum, v) => sum + Number(v ?? 0), 0);
    const gemsTotalEl = $(`p${p}-gems-total`);
    if (gemsTotalEl) gemsTotalEl.textContent = `总数:${gemsTotal}`;
    for (let c = 0; c < 6; c += 1) gems.appendChild(tokenChip(c, Number(gv[c] ?? 0)));
    const bonuses = $(`p${p}-bonuses`);
    bonuses.innerHTML = "";
    const bv = Array.isArray(pd.bonuses) ? pd.bonuses : [];
    for (let c = 0; c < 5; c += 1) {
      const chip = document.createElement("span");
      chip.className = `bonus-rect ${COLOR_NAMES[c]}`;
      chip.textContent = `${COLOR_LABELS[c]}:${Number(bv[c] ?? 0)}`;
      bonuses.appendChild(chip);
    }
  }
  renderReserved(0, "p0-reserved");
  renderReserved(1, "p1-reserved");
}

function renderTurnIndexBadge() {
  const badgeEl = $("turn-index-badge");
  if (!badgeEl) return;
  const roundIndex = Number(state.current?.public_state?.common?.round_index ?? 0);
  // round_index is ply-based; convert to human-friendly round number starting from 1.
  const roundNumber = Math.floor(Math.max(0, roundIndex) / 2) + 1;
  badgeEl.textContent = `第${roundNumber}回合`;
}

function renderTableau() {
  const root = $("tableau-root");
  root.innerHTML = "";
  const layout = document.createElement("div");
  layout.className = "tableau-grid-layout";
  const { visualTiers, logicalSlotMaps, hasTableauPayload } = buildTableauRenderModel();
  if (!hasTableauPayload) {
    const hint = document.createElement("div");
    hint.className = "muted";
    hint.textContent = "当前后端状态未返回桌面卡牌数据（tableau），请确认 splendor C++ 扩展是否为最新编译版本。";
    layout.appendChild(hint);
  }
  for (let t = 2; t >= 0; t -= 1) {
    const line = document.createElement("div");
    line.className = "tableau-tier-line";

    const reserveDeckActionId = 24 + t;
    const reserveDeckBtn = document.createElement("button");
    reserveDeckBtn.type = "button";
    reserveDeckBtn.className = "deck-op-btn deck-op-btn-left";
    reserveDeckBtn.textContent = "保留牌堆";
    reserveDeckBtn.disabled = !(canHumanControlCurrentTurn() && state.legalSet.has(reserveDeckActionId));
    reserveDeckBtn.addEventListener("click", () =>
      playAction(reserveDeckActionId).catch((e) => setMsg("ops-msg", e.message))
    );
    line.appendChild(reserveDeckBtn);

    const tier = document.createElement("div");
    tier.className = "tableau-tier";
    const title = document.createElement("div");
    title.className = "tableau-tier-title";
    const leftDeck = Number(state.current?.public_state?.game?.decks_remaining?.[t] ?? 0);
    const titleText = document.createElement("span");
    titleText.textContent = `Tier ${t + 1}（牌堆剩余 ${leftDeck}）`;
    title.appendChild(titleText);
    tier.appendChild(title);
    const row = document.createElement("div");
    row.className = "card-row";
    const cards = visualTiers[t] || [null, null, null, null];
    cards.forEach((card) => {
      const cid = getCardId(card);
      if (cid < 0) {
        const placeholder = document.createElement("div");
        placeholder.className = "dev-card dev-card-placeholder";
        row.appendChild(placeholder);
        return;
      }
      const canPlay = canHumanControlCurrentTurn();
      let buyEnabled = false;
      let reserveEnabled = false;
      const logicalSlot = logicalSlotMaps[t].get(cid);
      if (typeof logicalSlot === "number") {
        const buyId = t * 4 + logicalSlot;
        const reserveId = 12 + t * 4 + logicalSlot;
        buyEnabled = canPlay && state.legalSet.has(buyId);
        reserveEnabled = canPlay && state.legalSet.has(reserveId);
        const actions = [
          {
            label: "购买",
            disabled: !buyEnabled,
            onClick: () => playAction(buyId).catch((e) => setMsg("ops-msg", e.message)),
          },
          {
            label: "保留",
            disabled: !reserveEnabled,
            onClick: () => playAction(reserveId).catch((e) => setMsg("ops-msg", e.message)),
          },
        ];
        row.appendChild(renderCard(card, { actions }));
        return;
      }
      // Fallback: keep fixed button row to avoid visual jumping.
      row.appendChild(
        renderCard(card, {
          actions: [
            { label: "购买", disabled: true, onClick: null },
            { label: "保留", disabled: true, onClick: null },
          ],
        })
      );
    });
    tier.appendChild(row);
    line.appendChild(tier);
    layout.appendChild(line);
  }
  root.appendChild(layout);
}

function renderBankAndNobles() {
  const bankRow = $("bank-row");
  bankRow.innerHTML = "";
  const game = state.current?.public_state?.game || {};
  const bank = game.bank || [0, 0, 0, 0, 0, 0];
  const pendingReturns = Number(game.pending_returns ?? 0);
  const stage = String(game.stage || "normal");
  const pendingNobleSlots = Array.isArray(game.pending_noble_slots) ? game.pending_noble_slots.map((x) => Number(x)) : [];
  const canPlay = canHumanControlCurrentTurn();
  const inReturnMode = stage === "return_tokens";
  const inChooseNobleMode = stage === "choose_noble";
  if (!canPlay || inReturnMode || inChooseNobleMode) {
    state.gemPick.colors = [];
  }

  const gemPickColors = state.gemPick.colors;

  function hasAnyTakeAction() {
    for (let a = TAKE_THREE_OFFSET; a < CHOOSE_NOBLE_OFFSET; a += 1) {
      if (state.legalSet.has(a)) return true;
    }
    return false;
  }

  function canStartWithColor(color) {
    return resolveTakeActionId([color]) != null || hasTakeExtension([color]);
  }

  function resolveComboActionId(combos, offset, colors) {
    const sorted = [...colors].sort((a, b) => a - b);
    const comboIdx = combos.findIndex(
      (combo) => combo.length === sorted.length && combo.every((value, idx) => value === sorted[idx])
    );
    if (comboIdx < 0) return null;
    const actionId = offset + comboIdx;
    return state.legalSet.has(actionId) ? actionId : null;
  }

  function resolveTakeActionId(colors) {
    if (!Array.isArray(colors) || colors.length === 0) return null;
    if (colors.length === 2 && colors[0] === colors[1]) {
      const actionId = TAKE_TWO_SAME_OFFSET + colors[0];
      return state.legalSet.has(actionId) ? actionId : null;
    }
    const uniq = [...new Set(colors)];
    if (uniq.length !== colors.length) return null;
    if (uniq.length === 1) {
      const actionId = TAKE_ONE_OFFSET + uniq[0];
      return state.legalSet.has(actionId) ? actionId : null;
    }
    if (uniq.length === 2) {
      return resolveComboActionId(TAKE_TWO_DIFFERENT_COMBOS, TAKE_TWO_DIFFERENT_OFFSET, uniq);
    }
    if (colors.length === 3) {
      return resolveComboActionId(TAKE_THREE_COMBOS, TAKE_THREE_OFFSET, uniq);
    }
    return null;
  }

  function hasTakeExtension(colors) {
    if (!Array.isArray(colors) || colors.length === 0 || colors.length > 3) return false;
    if (resolveTakeActionId(colors) != null) return true;
    if (colors.length === 1) {
      const first = colors[0];
      if (resolveTakeActionId([first, first]) != null) return true;
      for (let c = 0; c < 5; c += 1) {
        if (c === first) continue;
        if (resolveTakeActionId([first, c]) != null) return true;
        for (let d = c + 1; d < 5; d += 1) {
          if (d === first) continue;
          if (resolveTakeActionId([first, c, d]) != null) return true;
        }
      }
      return false;
    }
    if (colors.length === 2 && colors[0] !== colors[1]) {
      for (let c = 0; c < 5; c += 1) {
        if (colors.includes(c)) continue;
        if (resolveTakeActionId([colors[0], colors[1], c]) != null) return true;
      }
      return false;
    }
    return false;
  }

  function canAppendColor(nextColor) {
    const picks = gemPickColors;
    if (picks.length >= 3) return false;
    if (picks.length === 0) return canStartWithColor(nextColor);
    const next = [...picks, nextColor];
    if (new Set(next).size !== next.length && !(next.length === 2 && next[0] === next[1])) {
      return false;
    }
    return hasTakeExtension(next);
  }

  function onBankGemClick(colorIdx) {
    if (!canPlay) return;
    if (inReturnMode) {
      const returnAction = RETURN_TOKEN_OFFSET + colorIdx;
      if (state.legalSet.has(returnAction)) {
        playAction(returnAction).catch((e) => setMsg("ops-msg", e.message));
      }
      return;
    }
    if (inChooseNobleMode) return;
    if (!hasAnyTakeAction()) return;
    if (!canAppendColor(colorIdx)) return;
    state.gemPick.colors = [...gemPickColors, colorIdx];
    renderBankAndNobles();
  }

  for (let c = 0; c < 6; c += 1) {
    const selectable =
      c < 5 &&
      canPlay &&
      !inChooseNobleMode &&
      (inReturnMode ? state.legalSet.has(RETURN_TOKEN_OFFSET + c) : canStartWithColor(c));
    const selected = gemPickColors.filter((x) => x === c).length > 0;
    bankRow.appendChild(
      tokenChip(c, Number(bank[c] ?? 0), {
        onClick: selectable ? () => onBankGemClick(c) : null,
        selected,
      })
    );
  }
  $("pending-return-msg").textContent =
    inReturnMode
      ? `当前需返还 ${pendingReturns} 枚宝石`
      : inChooseNobleMode
        ? "当前需要选择一位贵族。"
        : "";

  const pickStatus = $("gem-pick-status");
  const selectedText = gemPickColors.map((c) => COLOR_LABELS[c]).join(" + ");
  if (inReturnMode) {
    pickStatus.textContent = "返还模式：点击宝石堆返还 1 枚；可连续点击直到返还完成。";
  } else if (inChooseNobleMode) {
    pickStatus.textContent = "贵族选择模式：点击可选贵族完成本回合。";
  } else if (!canPlay) {
    pickStatus.textContent = "当前不是你的回合。";
  } else if (!hasAnyTakeAction()) {
    pickStatus.textContent = "当前无法拿宝石。";
  } else if (gemPickColors.length === 0) {
    pickStatus.textContent = "操作：点击宝石堆选择，可拿 1/2/3 种异色，或两枚同色，然后点确认。";
  } else {
    pickStatus.textContent = `已选择：${selectedText}（可确认或取消）`;
  }

  const confirmBtn = $("btn-gem-confirm");
  const cancelBtn = $("btn-gem-cancel");
  const takeActionId = resolveTakeActionId(gemPickColors);
  confirmBtn.disabled = !(canPlay && !inReturnMode && !inChooseNobleMode && takeActionId != null);
  confirmBtn.onclick = () => {
    if (takeActionId == null) return;
    playAction(takeActionId).catch((e) => setMsg("ops-msg", e.message));
  };
  cancelBtn.disabled = gemPickColors.length === 0 || inChooseNobleMode;
  cancelBtn.onclick = () => {
    state.gemPick.colors = [];
    renderBankAndNobles();
  };

  const passBtn = $("btn-pass-inline");
  const passLegal = state.legalSet.has(PASS_ACTION);
  passBtn.disabled = !(canPlay && passLegal && gemPickColors.length === 0);
  passBtn.onclick = () => playAction(PASS_ACTION).catch((e) => setMsg("ops-msg", e.message));

  const noblesRow = $("nobles-row");
  noblesRow.innerHTML = "";
  const nobles = game.nobles || [];
  for (const noble of nobles) {
    const card = document.createElement("div");
    card.className = "noble-card";
    const slot = Number(noble?.slot ?? -1);
    const selectable = canPlay && inChooseNobleMode && (Boolean(noble?.selectable) || pendingNobleSlots.includes(slot));
    if (selectable) card.classList.add("clickable");
    const submitNobleChoice = () =>
      playAction(CHOOSE_NOBLE_OFFSET + slot).catch((e) => setMsg("ops-msg", e.message));
    if (selectable) {
      card.addEventListener("click", () => submitNobleChoice());
      card.title = "点击选择该贵族";
    }
    const req = Array.isArray(noble.requirements) ? noble.requirements : [];
    const title = document.createElement("div");
    title.className = "noble-title";
    title.textContent = `贵族 ★${Number(noble.points ?? 3)}`;
    card.appendChild(title);
    const row = document.createElement("div");
    row.className = "cost-row";
    for (let c = 0; c < 5; c += 1) {
      const v = Number(req[c] ?? 0);
      if (v <= 0) continue;
      const pill = document.createElement("span");
      pill.className = `cost-gem ${COLOR_NAMES[c]}`;
      pill.textContent = String(v);
      row.appendChild(pill);
    }
    card.appendChild(row);
    if (selectable) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "card-op-btn";
      btn.textContent = "选择贵族";
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        submitNobleChoice();
      });
      card.appendChild(btn);
    }
    noblesRow.appendChild(card);
  }
}

async function refreshActions() {
  if (!state.sessionId) return [];
  return legalActionsStore.refreshActions();
}

async function refreshAll(aiOut = null) {
  if (aiOut && state.current) {
    const prevPayload = state.current;
    const plan = buildTransitionPlan(prevPayload, aiOut);
    const preRects = capturePlanRects(plan);
    state.current = aiOut;
    state.stateVersion = aiOut.state_version;
    await refreshActions();
    render();
    playTransitionAnimations(plan, preRects);
    return;
  }
  await turnRuntime.refreshState();
  await refreshActions();
  // refreshState() renders with old legal set; re-render after legal update
  // to avoid showing stale buy/reserve buttons on the new board.
  render();
}

async function playAction(actionId) {
  if (!state.sessionId || !state.current) return;
  if (!canHumanControlCurrentTurn()) return;
  if (!state.legalSet.has(Number(actionId))) return;
  const endpoint = state.forceMode ? "force-opponent-move" : "actions";
  const out = await api(`/api/v1/games/${state.sessionId}/${endpoint}`, "POST", {
    action_id: Number(actionId),
    state_version: state.stateVersion,
  });
  const prevPayload = state.current;
  const plan = buildTransitionPlan(prevPayload, out);
  const preRects = capturePlanRects(plan);
  state.current = out;
  state.stateVersion = out.state_version;
  await refreshActions();
  if (state.opsLocked) setOpsLocked(false);
  if (state.forceMode) {
    state.forceMode = false;
    setMsg("ops-msg", "已完成一次替对手落子");
  }
  state.gemPick.colors = [];
  render();
  playTransitionAnimations(plan, preRects);
  await turnRuntime.maybeAiRespondIfNeeded();
}

function render() {
  renderTopWithHint({
    state,
    hintPanel,
    byId: $,
    computeScoreInfo: getScoreInfo,
    drawTurnText: "对局结束：平局",
    drawMessageText: "本局已结束，结果为平局。",
  });
  renderTurnIndexBadge();
  renderBankAndNobles();
  renderTableau();
  renderPlayers();
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
  refreshAfterAi: refreshAll,
  mapPolicy,
  gameLabelForStart: (humanPlayer, aiLabel, sessionId) => `已开局 ${sessionId}，你是玩家${humanPlayer}，难度=${aiLabel}`,
});

const legalActionsStore = createLegalActionsStore({
  api,
  endpointBuilder: () => `/api/v1/games/${state.sessionId}/legal-actions`,
  getCurrentStateVersion: () => state.stateVersion,
  onApply: ({ actions }) => {
    state.legalActions = actions || [];
    state.legalSet = new Set(state.legalActions.map((x) => Number(x.action_id)));
  },
});

async function boot() {
  resetTableauViewState();
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
      storageKey: "dino_splendor_debug_zoom_percent",
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

