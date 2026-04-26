export function createLegalActionsStore({
  api,
  endpointBuilder,
  getCurrentStateVersion = null,
  onApply,
  onLoadingChange = null,
  extractActions = (data) => data?.actions || [],
  extractStateVersion = (data) => Number(data?.state_version ?? -1),
}) {
  let requestSeq = 0;

  async function refreshActions() {
    const reqSeq = ++requestSeq;
    if (onLoadingChange) onLoadingChange(true);
    try {
      const data = await api(endpointBuilder());
      if (reqSeq !== requestSeq) return [];
      if (getCurrentStateVersion) {
        const expected = Number(getCurrentStateVersion());
        const got = Number(extractStateVersion(data));
        if (Number.isFinite(expected) && Number.isFinite(got) && expected !== got) {
          return [];
        }
      }
      const actions = extractActions(data);
      onApply({ actions, data });
      return actions;
    } finally {
      if (onLoadingChange) onLoadingChange(false);
    }
  }

  return { refreshActions };
}
