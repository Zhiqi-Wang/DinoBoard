function normalizeUrl(url) {
  try {
    return new URL(url, window.location.href).toString();
  } catch (_) {
    return String(url || "");
  }
}

async function isReachable(url, timeoutMs = 1200) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    // no-cors is enough for connectivity check across local ports.
    await fetch(url, { method: "GET", mode: "no-cors", cache: "no-store", signal: controller.signal });
    return true;
  } catch (_) {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export function initGameSelector(config) {
  const { selectEl, games = [], currentGameId = "" } = config || {};
  if (!selectEl) return;

  let currentGames = Array.isArray(games) ? [...games] : [];

  function renderOptions() {
    selectEl.innerHTML = "";
    for (const g of currentGames) {
      const opt = document.createElement("option");
      opt.value = String(g?.id || "");
      opt.textContent = String(g?.label || g?.id || "");
      if (opt.value === String(currentGameId)) {
        opt.selected = true;
      }
      selectEl.appendChild(opt);
    }
  }
  renderOptions();

  selectEl.addEventListener("change", async () => {
    const id = String(selectEl.value || "");
    const target = currentGames.find((g) => String(g?.id || "") === id);
    if (!target || !target.url) return;
    const next = normalizeUrl(target.url);
    const cur = window.location.href;
    if (next === cur) return;

    const ok = await isReachable(next);
    if (!ok) {
      // Keep current page selected when target service is down.
      selectEl.value = String(currentGameId);
      window.alert("目标游戏服务未启动，请先启动对应 debug_service 后再切换。");
      return;
    }
    window.location.href = next;
  });

  // In hub mode, /api/games is authoritative; in single-game mode this request fails and fallback remains.
  fetch("/api/games", { method: "GET", cache: "no-store" })
    .then((resp) => (resp.ok ? resp.json() : null))
    .then((data) => {
      if (!Array.isArray(data) || data.length === 0) return;
      const normalized = data
        .map((item) => ({
          id: String(item?.id || "").trim(),
          label: String(item?.label || item?.id || "").trim(),
          url: String(item?.url || "").trim(),
        }))
        .filter((item) => item.id && item.url);
      if (normalized.length === 0) return;
      currentGames = normalized;
      renderOptions();
    })
    .catch(() => {});
}

