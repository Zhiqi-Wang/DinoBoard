export async function rewindToTurn(config) {
  const {
    api,
    sessionId,
    getStateVersion,
    getCurrentPlayer,
    refreshAll,
    targetPlayer,
    maxSteps = 256,
  } = config;

  if (!sessionId) return false;
  const startVersion = Number(getStateVersion ? getStateVersion() : 0);
  let guard = 0;
  let stepped = false;

  while (Number(getStateVersion()) > 0 && guard < maxSteps) {
    await api(`/api/v1/games/${sessionId}/step-back`, "POST", {
      state_version: Number(getStateVersion()),
    });
    await refreshAll();
    stepped = true;
    if (Number(getStateVersion()) < startVersion && Number(getCurrentPlayer()) === Number(targetPlayer)) {
      return true;
    }
    guard += 1;
  }

  return stepped && Number(getStateVersion()) < startVersion && Number(getCurrentPlayer()) === Number(targetPlayer);
}

