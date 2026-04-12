export function renderSelectedPick({ selectedPick, selectedPickEl, tileClass }) {
  if (!selectedPickEl) return;
  if (!selectedPick) {
    selectedPickEl.innerHTML = '<span class="muted">未选择来源砖</span>';
    return;
  }
  selectedPickEl.innerHTML = `
    <span class="muted">已选：</span>
    <span class="tile ${tileClass(selectedPick.color)}">${selectedPick.color}</span>
    <span class="mono">${selectedPick.source}[${selectedPick.source_idx}]</span>
  `;
}

export function renderTargetRow({
  targetRowEl,
  selectedPick,
  actionIndex,
  opTargetLines,
}) {
  if (!targetRowEl) return;
  targetRowEl.innerHTML = "";
  const targets = selectedPick ? actionIndex.get(selectedPick.sourceKey) : null;
  for (const t of opTargetLines) {
    const chip = document.createElement("span");
    const action = targets?.get(t);
    chip.className = `target-chip ${action ? "enabled" : "disabled"}`;
    chip.textContent = t === -1 ? "floor" : `line ${t}`;
    targetRowEl.appendChild(chip);
  }
}
