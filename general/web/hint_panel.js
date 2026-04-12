function asPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  const v = Math.max(0, Math.min(1, value));
  return `${(v * 100).toFixed(1)}%`;
}

export function createHintPanel(config) {
  const {
    turnEl,
    messageEl,
    scoreEl,
    winrateEl,
    suggestionEl = null,
    introMessage = "",
    unknownWinrateText = "--",
    unknownScoreText = "--",
    defaultSuggestionText = "AI 提示：--",
    formatTurn = (player) => `当前轮到：玩家${player}`,
    formatScore = (scoreInfo) => `当前分数：${JSON.stringify(scoreInfo)}`,
    formatOpponentMove = null,
    formatSuggestedMove = null,
  } = config;

  let introShown = false;

  function setTurn(currentPlayer) {
    if (turnEl) turnEl.textContent = formatTurn(currentPlayer);
  }

  function showIntro() {
    if (messageEl) messageEl.textContent = introMessage;
    introShown = true;
  }

  function showOpponentMove(moveInfo) {
    if (!messageEl) return;
    if (formatOpponentMove) {
      messageEl.textContent = formatOpponentMove(moveInfo);
      return;
    }
    messageEl.textContent = "对手已完成一步";
  }

  function showSuggestedMove(moveInfo) {
    if (!suggestionEl) return;
    if (formatSuggestedMove) {
      suggestionEl.textContent = formatSuggestedMove(moveInfo);
      return;
    }
    if (formatOpponentMove) {
      suggestionEl.textContent = `AI 提示：${formatOpponentMove(moveInfo)}`;
      return;
    }
    const aid = Number(moveInfo?.action_id ?? -1);
    suggestionEl.textContent = aid >= 0 ? `AI 提示：推荐 action_id=${aid}` : defaultSuggestionText;
  }

  function clearSuggestedMove() {
    if (!suggestionEl) return;
    suggestionEl.textContent = defaultSuggestionText;
  }

  function setOpponentWinrate(winrate) {
    if (!winrateEl) return;
    if (typeof winrate !== "number" || Number.isNaN(winrate)) {
      winrateEl.textContent = `对手预估胜率：${unknownWinrateText}`;
      return;
    }
    winrateEl.textContent = `对手预估胜率：${asPercent(winrate)}`;
  }

  function setScore(scoreInfo) {
    if (!scoreEl) return;
    if (scoreInfo == null) {
      scoreEl.textContent = `当前分数：${unknownScoreText}`;
      return;
    }
    scoreEl.textContent = formatScore(scoreInfo);
  }

  function ensureIntroIfNeeded() {
    if (!introShown) showIntro();
  }

  return {
    setTurn,
    showIntro,
    showOpponentMove,
    setScore,
    setOpponentWinrate,
    showSuggestedMove,
    clearSuggestedMove,
    ensureIntroIfNeeded,
  };
}

