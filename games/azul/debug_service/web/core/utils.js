export function tileClass(c) {
  const m = { B: "tile-B", Y: "tile-Y", R: "tile-R", K: "tile-K", W: "tile-W", F: "tile-F" };
  return m[c] || "tile-X";
}

export function renderTileList(tiles) {
  if (!tiles || tiles.length === 0) return '<span class="muted">-</span>';
  return tiles
    .map((c) => `<span class="tile ${tileClass(c)}" title="${c}">${c}</span>`)
    .join("");
}

export function wallColorAt(row, col) {
  const colors = ["B", "Y", "R", "K", "W"];
  const idx = ((col - row) % 5 + 5) % 5;
  return colors[idx];
}

export function pickKey(source, sourceIdx, color) {
  return `${source}|${sourceIdx}|${color}`;
}

export function buildActionIndex(actions) {
  const map = new Map();
  for (const a of actions) {
    const m = a.move;
    const pkey = pickKey(m.source, m.source_idx, m.color);
    if (!map.has(pkey)) map.set(pkey, new Map());
    map.get(pkey).set(m.target_line, a);
  }
  return map;
}
