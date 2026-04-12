import { initGameSelector } from "./game_selector.js";
import { initZoomControls } from "./zoom_controls.js";
import { bindSidebarControls } from "./sidebar_controls.js";

export function setupTwoPlayerPageControls({
  gameSelector,
  zoomControls,
  sidebarControls,
}) {
  initGameSelector(gameSelector);
  initZoomControls(zoomControls);
  bindSidebarControls(sidebarControls);
}

export async function tryAutoStart({ start, onError }) {
  try {
    await start();
  } catch (e) {
    if (onError) onError(e);
  }
}
