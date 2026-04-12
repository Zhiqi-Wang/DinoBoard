import { rewindToTurn } from "./turn_ops.js";

function ensureGameOverModal() {
  let backdrop = document.getElementById("general-gameover-backdrop");
  if (backdrop) {
    return {
      backdrop,
      textEl: document.getElementById("general-gameover-text"),
    };
  }
  backdrop = document.createElement("div");
  backdrop.id = "general-gameover-backdrop";
  backdrop.className = "general-modal-backdrop hidden";

  const panel = document.createElement("div");
  panel.className = "general-modal-panel";
  const title = document.createElement("div");
  title.className = "general-modal-title";
  title.textContent = "对局结束";
  const text = document.createElement("div");
  text.id = "general-gameover-text";
  text.className = "general-modal-text";
  text.textContent = "--";
  const okBtn = document.createElement("button");
  okBtn.type = "button";
  okBtn.className = "general-modal-ok-btn";
  okBtn.textContent = "知道了";
  okBtn.addEventListener("click", () => {
    backdrop.classList.add("hidden");
  });

  panel.append(title, text, okBtn);
  backdrop.appendChild(panel);
  document.body.appendChild(backdrop);
  return { backdrop, textEl: text };
}

export function renderTopWithHint({ state, hintPanel, byId, computeScoreInfo, drawTurnText, drawMessageText }) {
  const common = state.current?.public_state?.common;
  if (!common) return;
  if (common.is_terminal) {
    if (common.winner == null) {
      byId("hint-turn-line").textContent = drawTurnText;
      byId("hint-message-line").textContent = drawMessageText;
    } else {
      byId("hint-turn-line").textContent = `对局结束：玩家${common.winner}获胜`;
      byId("hint-message-line").textContent = `本局已结束，玩家${common.winner}获胜。`;
    }
  } else {
    hintPanel.setTurn(common.current_player);
  }
  hintPanel.setScore(computeScoreInfo(state.current?.public_state));
}

export function createTwoPlayerTurnRuntime({
  state,
  api,
  hintPanel,
  setMsg,
  setOpsLocked,
  canHumanControlCurrentTurn,
  render,
  refreshActions,
  refreshAfterAi,
  mapPolicy,
  gameLabelForStart,
  onStartReset,
  onUndoReset,
  onForceReset,
  isHumanTurn,
  resolveForceTargetPlayer,
  guardLimit = 16,
}) {
  let shownGameOverStateKey = "";

  function syncGameOverModal() {
    const common = state.current?.public_state?.common;
    const { backdrop, textEl } = ensureGameOverModal();
    if (!common?.is_terminal) {
      backdrop.classList.add("hidden");
      shownGameOverStateKey = "";
      return;
    }
    const stateKey = `${state.sessionId || "no_session"}:${Number(state.stateVersion ?? -1)}`;
    if (shownGameOverStateKey === stateKey) return;
    shownGameOverStateKey = stateKey;
    if (common.winner == null) {
      textEl.textContent = "本局结果：平局";
    } else if (Number(common.winner) === Number(state.humanPlayer)) {
      textEl.textContent = `本局结果：你获胜（玩家${common.winner}）`;
    } else {
      textEl.textContent = `本局结果：玩家${common.winner}获胜`;
    }
    backdrop.classList.remove("hidden");
  }

  async function refreshState() {
    if (!state.sessionId) return;
    const payload = await api(`/api/v1/games/${state.sessionId}/state`);
    state.current = payload;
    state.stateVersion = payload.state_version;
    render();
    syncGameOverModal();
  }

  async function refreshAll() {
    await refreshState();
    if (refreshActions) {
      await refreshActions();
    }
  }

  async function maybeAiRespondIfNeeded() {
    if (!state.sessionId || !state.current) return;
    let guard = 0;
    while (guard < guardLimit) {
      const common = state.current.public_state.common;
      if (common.is_terminal) {
        syncGameOverModal();
        return;
      }
      if (state.forceMode) return;
      if (isHumanTurn ? isHumanTurn() : common.current_player === state.humanPlayer) return;
      const aiOut = await api(`/api/v1/games/${state.sessionId}/ai-move`, "POST", {
        state_version: state.stateVersion,
        engine: state.aiPolicy.engine,
        simulations: state.aiPolicy.simulations,
        time_budget_ms: state.aiPolicy.time_budget_ms || 0,
        temperature: state.aiPolicy.temperature,
        model_path: state.aiPolicy.model_path || null,
        search_options: state.aiPolicy.search_options || null,
      });
      if (aiOut?.event) {
        hintPanel.showOpponentMove(aiOut.event);
        hintPanel.setOpponentWinrate(aiOut.event.estimated_winrate);
      } else {
        hintPanel.setOpponentWinrate(null);
      }
      if (refreshAfterAi) {
        await refreshAfterAi(aiOut);
      } else {
        await refreshState();
      }
      guard += 1;
    }
    setMsg("ops-msg", "AI 连续动作超出上限，请检查规则流程");
  }

  async function handleStart() {
    let humanPlayer = 0;
    if (state.sideMode === "second") {
      humanPlayer = 1;
    } else if (state.sideMode === "random") {
      humanPlayer = Math.random() < 0.5 ? 0 : 1;
    }
    const out = await api("/api/v1/games", "POST", { seed: null, human_player: humanPlayer });
    state.sessionId = out.session_id;
    state.current = out;
    state.stateVersion = out.state_version;
    state.humanPlayer = humanPlayer;
    state.forceMode = false;
    shownGameOverStateKey = "";
    setOpsLocked(false);
    if (onStartReset) onStartReset();
    hintPanel.showIntro();
    hintPanel.setOpponentWinrate(null);
    if (hintPanel.clearSuggestedMove) {
      hintPanel.clearSuggestedMove();
    }
    if (refreshActions) {
      await refreshActions();
    }
    const label = gameLabelForStart ? gameLabelForStart(humanPlayer, state.aiPolicy.label, out.session_id) : "";
    if (label) setMsg("start-msg", label);
    render();
    syncGameOverModal();
    await maybeAiRespondIfNeeded();
  }

  async function handleUndo() {
    if (!state.sessionId || state.opsLocked) return;
    const ok = await rewindToTurn({
      api,
      sessionId: state.sessionId,
      getStateVersion: () => state.stateVersion,
      getCurrentPlayer: () => state.current?.public_state?.common?.current_player,
      refreshAll,
      targetPlayer: state.humanPlayer,
    });
    if (!ok) {
      setMsg("ops-msg", "无法回退到上一手玩家可操作局面");
      return;
    }
    state.forceMode = false;
    setOpsLocked(true);
    if (onUndoReset) onUndoReset();
    setMsg("ops-msg", "已悔棋：待你操作后恢复按钮");
    render();
  }

  async function handleForce() {
    if (!state.sessionId || state.opsLocked || !state.current) return;
    const common = state.current.public_state.common;
    if (common.is_terminal) {
      setMsg("ops-msg", "终局不可替对手落子");
      return;
    }
    const opponent = resolveForceTargetPlayer ? resolveForceTargetPlayer() : 1 - state.humanPlayer;
    const ok = await rewindToTurn({
      api,
      sessionId: state.sessionId,
      getStateVersion: () => state.stateVersion,
      getCurrentPlayer: () => state.current?.public_state?.common?.current_player,
      refreshAll,
      targetPlayer: opponent,
    });
    if (!ok) {
      setMsg("ops-msg", "无法回退到上一手对手可操作局面");
      return;
    }
    state.forceMode = true;
    setOpsLocked(true);
    if (onForceReset) onForceReset();
    setMsg("ops-msg", "已回退到对手回合：请在棋盘点击一步");
    render();
  }

  async function handleHint() {
    if (!state.sessionId || !state.current) return;
    const common = state.current.public_state.common;
    if (common.is_terminal) {
      setMsg("ops-msg", "终局无可用提示");
      return;
    }
    const out = await api(`/api/v1/games/${state.sessionId}/ai-hint`, "POST", {
      state_version: state.stateVersion,
      engine: state.aiPolicy.engine,
      simulations: state.aiPolicy.simulations,
      time_budget_ms: state.aiPolicy.time_budget_ms || 0,
      temperature: state.aiPolicy.temperature,
      model_path: state.aiPolicy.model_path || null,
      search_options: state.aiPolicy.search_options || null,
    });
    if (out?.event) {
      hintPanel.showSuggestedMove(out.event);
    }
    state.current = out;
    state.stateVersion = out.state_version;
    render();
    syncGameOverModal();
  }

  function bindDifficulty(mode, policyFromGeneral) {
    state.difficultyMode = mode;
    state.aiPolicy = mapPolicy(mode, policyFromGeneral);
  }

  return {
    refreshState,
    refreshAll,
    maybeAiRespondIfNeeded,
    handleStart,
    handleUndo,
    handleForce,
    handleHint,
    bindDifficulty,
    canHumanControlCurrentTurn,
  };
}
