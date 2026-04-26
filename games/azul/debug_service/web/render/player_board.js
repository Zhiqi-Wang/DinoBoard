export function renderPlayerBoard({
  rootEl,
  playerData,
  ownerPid,
  currentPlayer,
  interactiveActor,
  selectedPick,
  allowedTargets,
  tileClass,
  renderTileList,
  wallColorAt,
  floorPenalties,
  onTargetClick,
}) {
  if (!rootEl) return;
  const lines = playerData.pattern_lines || [];
  const wall = playerData.wall || [];
  const floor = playerData.floor || [];

  const pattern = document.createElement("div");
  pattern.className = "pattern-lines";
  const ownerActive = interactiveActor === ownerPid;
  for (let r = 0; r < 5; r++) {
    const row = document.createElement("div");
    const clickable = ownerActive && selectedPick && allowedTargets.has(r);
    row.className = `pattern-row target-slot ${clickable ? "active" : "disabled"}`;
    row.dataset.targetLine = String(r);
    if (clickable && onTargetClick) {
      row.onclick = () => onTargetClick(r);
    }
    const label = document.createElement("span");
    label.className = "pattern-label";
    label.textContent = `line ${r}`;
    row.appendChild(label);
    const line = lines[r] || [];
    const cap = r + 1;
    const empties = cap - line.length;
    for (let i = 0; i < empties; i++) {
      const slot = document.createElement("span");
      slot.className = "slot";
      row.appendChild(slot);
    }
    for (const c of line) {
      const tile = document.createElement("span");
      tile.className = `tile ${tileClass(c)}`;
      tile.textContent = c;
      row.appendChild(tile);
    }
    pattern.appendChild(row);
  }

  const wallGrid = document.createElement("div");
  wallGrid.className = "wall-grid";
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const filled = !!(wall[r] && wall[r][c]);
      const cell = document.createElement("span");
      const hintColor = wallColorAt(r, c);
      cell.className = `wall-cell ${filled ? "wall-filled" : `wall-empty wall-hint-${hintColor}`}`;
      if (filled) {
        const tile = document.createElement("span");
        tile.className = `tile wall-tile ${tileClass(hintColor)}`;
        cell.appendChild(tile);
      }
      wallGrid.appendChild(cell);
    }
  }

  const floorRow = document.createElement("div");
  const floorClickable = ownerActive && selectedPick && allowedTargets.has(-1);
  floorRow.className = `floor-row target-slot ${floorClickable ? "active" : "disabled"}`;
  floorRow.dataset.targetLine = "-1";
  if (floorClickable && onTargetClick) {
    floorRow.onclick = () => onTargetClick(-1);
  }
  const floorMain = document.createElement("div");
  floorMain.className = "floor-main";
  const floorTail = document.createElement("div");
  floorTail.className = "floor-tail";
  for (let i = 0; i < 7; i++) {
    const slot = document.createElement("span");
    slot.className = "floor-slot";
    slot.dataset.ownerPid = String(ownerPid);
    slot.dataset.floorIndex = String(i);
    const base = document.createElement("span");
    base.className = "tile tile-empty floor-base";
    slot.appendChild(base);
    const hasTile = i < floor.length;
    if (hasTile) {
      const t = document.createElement("span");
      t.className = `tile floor-tile ${tileClass(floor[i])}`;
      t.textContent = floor[i];
      slot.appendChild(t);
    } else {
      const pen = document.createElement("span");
      pen.className = "pen pen-badge";
      pen.textContent = String(floorPenalties[i]);
      slot.appendChild(pen);
    }
    if (i < 5) {
      floorMain.appendChild(slot);
    } else {
      floorTail.appendChild(slot);
    }
  }
  floorRow.appendChild(floorMain);
  floorRow.appendChild(floorTail);

  rootEl.innerHTML = "";
  rootEl.appendChild(pattern);
  rootEl.appendChild(wallGrid);
  rootEl.appendChild(floorRow);
}

export function getTargetElementForMove({ targetLine, currentState, getById }) {
  if (targetLine === -1) {
    const currentPlayer = currentState?.public_state?.common?.current_player;
    if (currentPlayer === undefined || currentPlayer === null) return null;
    const floorLen = currentState?.public_state?.game?.players?.[currentPlayer]?.floor?.length || 0;
    const idx = Math.max(0, Math.min(6, floorLen));
    return document.querySelector(
      `#p${currentPlayer}-board .floor-slot[data-owner-pid="${currentPlayer}"][data-floor-index="${idx}"]`
    );
  }
  const currentPlayer = currentState?.public_state?.common?.current_player;
  if (currentPlayer === undefined || currentPlayer === null) return null;
  return getById(`p${currentPlayer}-title`);
}
