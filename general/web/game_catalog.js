export const GAME_CATALOG = [
  { id: "azul", label: "花砖物语", url: "/games/azul/" },
  { id: "quoridor", label: "步步为营", url: "/games/quoridor/" },
  { id: "splendor", label: "璀璨宝石", url: "/games/splendor/" },
  { id: "tictactoe", label: "井字棋", url: "/games/tictactoe/" },
];

let cachedCatalog = GAME_CATALOG;

function normalizeCatalog(raw) {
  if (!Array.isArray(raw)) return null;
  const normalized = raw
    .map((item) => ({
      id: String(item?.id || "").trim(),
      label: String(item?.label || item?.id || "").trim(),
      url: String(item?.url || "").trim(),
    }))
    .filter((item) => item.id && item.url);
  return normalized.length > 0 ? normalized : null;
}

export async function loadGameCatalog() {
  try {
    const resp = await fetch("/api/games", { method: "GET", cache: "no-store" });
    if (resp.ok) {
      const data = await resp.json();
      const normalized = normalizeCatalog(data);
      if (normalized) {
        cachedCatalog = normalized;
      }
    }
  } catch (_) {
    // single-game mode may not expose /api/games; fallback to static list.
  }
  return cachedCatalog;
}

export function currentGameIdFromPath(pathname, games = cachedCatalog) {
  const path = String(pathname || "");
  for (const game of games || []) {
    if (path.includes(`/games/${game.id}`)) return game.id;
  }
  return "";
}
