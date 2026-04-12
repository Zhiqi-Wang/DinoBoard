import { getTargetElementForMove } from "../render/player_board.js";

export function createMoveFlow(config) {
  const {
    state,
    getById,
    tileClass,
    api,
    refreshAllFromPayload,
    clearPickSelection,
    maybeAiRespondIfNeeded,
    onPlayerOperated,
    setOpsMsg,
  } = config;

  function getTargetElement(targetLine) {
    return getTargetElementForMove({
      targetLine,
      currentState: state.current,
      getById,
    });
  }

  async function animateTileMove(sourceEl, targetEl, color) {
    if (!sourceEl || !targetEl) return;
    const layer = getById("tile-layer");
    if (!layer) return;
    const src = sourceEl.getBoundingClientRect();
    const dst = targetEl.getBoundingClientRect();
    const tile = document.createElement("div");
    tile.className = `anim-tile ${tileClass(color)}`;
    tile.textContent = color;
    tile.style.left = `${src.left + src.width / 2 - 13}px`;
    tile.style.top = `${src.top + src.height / 2 - 13}px`;
    layer.appendChild(tile);
    await new Promise((resolve) => requestAnimationFrame(resolve));
    tile.style.transform = `translate(${dst.left - src.left}px, ${dst.top - src.top}px) scale(0.9)`;
    tile.style.opacity = "0.25";
    await new Promise((resolve) => setTimeout(resolve, 380));
    tile.remove();
  }

  function showSettlementAnimation(event) {
    const scoring = event?.apply_result?.round_scoring;
    if (!scoring?.per_player) return;
    for (const p of scoring.per_player) {
      const host = getById(`p${p.player}-title`);
      if (!host) continue;
      const pop = document.createElement("span");
      pop.className = "score-pop";
      const delta = (p.round_gain || 0) + (p.floor_penalty || 0);
      pop.textContent = `${delta >= 0 ? "+" : ""}${delta}`;
      host.appendChild(pop);
      requestAnimationFrame(() => pop.classList.add("show"));
      setTimeout(() => pop.remove(), 1200);
    }
  }

  async function playActionWithAnimation(action) {
    const move = action.move;
    const sourceEl = state.selectedPick?.el || document.querySelector(
      `.tile-pick[data-source="${move.source}"][data-source-idx="${move.source_idx}"][data-color="${move.color}"]`
    );
    const targetEl = getTargetElement(move.target_line);
    if (sourceEl && targetEl) {
      await animateTileMove(sourceEl, targetEl, move.color);
    }
    const common = state.current?.public_state?.common;
    const useForce = state.forceMode && common && common.current_player !== state.humanPlayer;
    const endpoint = useForce
      ? `/api/v1/games/${state.sessionId}/force-opponent-move`
      : `/api/v1/games/${state.sessionId}/actions`;
    const resp = await api(endpoint, "POST", {
      action_id: action.action_id,
      state_version: state.stateVersion,
    });
    await refreshAllFromPayload(resp);
    showSettlementAnimation(resp.event);
    if (useForce) state.forceMode = false;
    if (onPlayerOperated) onPlayerOperated();
    clearPickSelection();
    await maybeAiRespondIfNeeded();
  }

  async function playSelectedTarget(targetLine) {
    if (!state.selectedPick) return;
    const targets = state.actionIndex.get(state.selectedPick.sourceKey);
    const action = targets?.get(targetLine);
    if (!action) {
      setOpsMsg("该目标槽当前不可落子");
      return;
    }
    await playActionWithAnimation(action);
  }

  return { playSelectedTarget };
}
