import { getDifficultyPolicy } from "./difficulty_controls.js";

export function byId(id) {
  return document.getElementById(id);
}

export function setTextById(id, msg) {
  const el = byId(id);
  if (el) el.textContent = msg || "";
}

export function mapDifficultyToPolicy(mode, policyFromGeneral, simsMap = null) {
  const basePolicy = getDifficultyPolicy(mode);
  const base = {
    engine: String(basePolicy?.engine || "netmcts"),
    simulations: Number(basePolicy?.simulations || 0),
    temperature: Number(basePolicy?.temperature || 0),
    label: String(basePolicy?.label || "体验"),
  };
  if (base.engine === "heuristic") {
    return { ...base, time_budget_ms: 0 };
  }
  const overrideSims = Number((simsMap || {})[mode]);
  const generalSims = Number(policyFromGeneral?.simulations);
  const baseSims = Number(base.simulations || 10);
  const sims = Number.isFinite(overrideSims) && overrideSims > 0
    ? overrideSims
    : Number.isFinite(generalSims) && generalSims > 0
      ? generalSims
      : baseSims;
  return {
    engine: base.engine,
    simulations: sims,
    temperature: base.temperature,
    time_budget_ms: 0,
    label: base.label,
  };
}

export function canHumanControlTurn(current, humanPlayer, forceMode) {
  const common = current?.public_state?.common;
  if (!common || common.is_terminal) return false;
  if (common.current_player === humanPlayer) return true;
  return !!forceMode && common.current_player !== humanPlayer;
}

export function setButtonsDisabled(ids, disabled) {
  for (const id of ids) {
    const btn = byId(id);
    if (btn) btn.disabled = !!disabled;
  }
}

export function scoreFromPublicState(publicState) {
  const scores = Array.isArray(publicState?.common?.scores) ? publicState.common.scores : [0, 0];
  return { p0: Number(scores[0] || 0), p1: Number(scores[1] || 0) };
}
