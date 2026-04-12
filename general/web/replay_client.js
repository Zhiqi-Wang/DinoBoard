import { api } from "./api_client.js";

export function prettyJson(obj) {
  return JSON.stringify(obj, null, 2);
}

export async function fetchReplay(sessionId) {
  return api(`/api/v1/games/${sessionId}/replay`);
}
