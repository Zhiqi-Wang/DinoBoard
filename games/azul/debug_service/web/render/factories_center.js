export function renderFactories({
  factories,
  rootEl,
  actionIndex,
  isActionIndexReady,
  canPickSource,
  pickKey,
  tileClass,
  setOpsMsg,
  onPick,
}) {
  if (!rootEl) return;
  rootEl.innerHTML = "";
  const w = rootEl.clientWidth || 560;
  const h = rootEl.clientHeight || 430;
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(w, h) * 0.34;

  factories.forEach((f, idx) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * idx) / 5;
    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle);
    const div = document.createElement("div");
    div.className = "factory";
    div.style.left = `${x}px`;
    div.style.top = `${y}px`;

    const title = document.createElement("strong");
    title.className = "factory-title";
    title.textContent = `F${idx}`;
    div.appendChild(title);

    const disc = document.createElement("div");
    disc.className = "factory-disc";
    const tilesWrap = document.createElement("div");
    tilesWrap.className = "factory-tiles";
    if (!f || f.length === 0) {
      tilesWrap.innerHTML = '<span class="muted">-</span>';
    } else {
      const counts = new Map();
      for (const c of f) counts.set(c, (counts.get(c) || 0) + 1);
      for (const c of f) {
        tilesWrap.appendChild(makePickTile({
          source: "factory",
          sourceIdx: idx,
          color: c,
          count: counts.get(c) || 1,
          showBadge: false,
          actionIndex,
          isActionIndexReady,
          canPickSource,
          pickKey,
          tileClass,
          setOpsMsg,
          onPick,
        }));
      }
    }
    disc.appendChild(tilesWrap);
    div.appendChild(disc);
    rootEl.appendChild(div);
  });
}

export function renderCenter({
  tiles,
  firstPlayerTokenInCenter,
  rootEl,
  actionIndex,
  isActionIndexReady,
  canPickSource,
  pickKey,
  tileClass,
  setOpsMsg,
  onPick,
}) {
  if (!rootEl) return;
  rootEl.innerHTML = "";
  const hasTiles = !!(tiles && tiles.length > 0);
  const hasFirstToken = !!firstPlayerTokenInCenter;
  if (!hasTiles && !hasFirstToken) {
    rootEl.innerHTML = '<span class="muted">-</span>';
    return;
  }
  const row = document.createElement("div");
  row.className = "tile-row";
  const counts = new Map();
  for (const c of tiles || []) counts.set(c, (counts.get(c) || 0) + 1);
  [...counts.entries()].forEach(([color, count]) => {
    row.appendChild(makePickTile({
      source: "center",
      sourceIdx: -1,
      color,
      count,
      showBadge: true,
      actionIndex,
      isActionIndexReady,
      canPickSource,
      pickKey,
      tileClass,
      setOpsMsg,
      onPick,
    }));
  });
  if (hasFirstToken) {
    const token = document.createElement("span");
    token.className = `tile ${tileClass("F")}`;
    token.textContent = "F";
    token.title = "先手标记";
    row.appendChild(token);
  }
  rootEl.appendChild(row);
}

function makePickTile({
  source,
  sourceIdx,
  color,
  count,
  showBadge,
  actionIndex,
  isActionIndexReady,
  canPickSource,
  pickKey,
  tileClass,
  setOpsMsg,
  onPick,
}) {
  const btn = document.createElement("button");
  btn.className = `tile tile-pick ${tileClass(color)}`;
  btn.dataset.source = source;
  btn.dataset.sourceIdx = String(sourceIdx);
  btn.dataset.color = color;
  btn.textContent = color;
  btn.title = `${source}[${sourceIdx}] ${color} x${count}`;
  if (showBadge) {
    const badge = document.createElement("span");
    badge.className = "tile-count";
    badge.textContent = String(count);
    btn.appendChild(badge);
  }
  btn.onclick = () => {
    if (typeof canPickSource === "function" && !canPickSource()) {
      setOpsMsg("当前不是你的可操作回合");
      return;
    }
    if (typeof isActionIndexReady === "function" && !isActionIndexReady()) {
      setOpsMsg("动作列表刷新中，请稍后再试");
      return;
    }
    const key = pickKey(source, sourceIdx, color);
    if (!actionIndex || !actionIndex.has(key)) {
      setOpsMsg("该来源砖色当前无可执行目标");
      return;
    }
    if (onPick) {
      onPick({
        source,
        source_idx: sourceIdx,
        color,
        sourceKey: key,
        el: btn,
      });
    }
  };
  return btn;
}
