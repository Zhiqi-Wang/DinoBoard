function detectMountBase() {
  const p = String(window.location.pathname || "");
  const m = p.match(/^\/games\/[^/]+/);
  return m ? m[0] : "";
}

function normalizePath(path) {
  const raw = String(path || "");
  if (/^https?:\/\//i.test(raw)) return raw;
  const base = detectMountBase();
  if (!base) return raw;
  if (raw.startsWith("/api/") || raw.startsWith("/web/") || raw.startsWith("/general-web/")) {
    return `${base}${raw}`;
  }
  return raw;
}

export async function api(path, method = "GET", body = null) {
  const options = { method, headers: {} };
  if (body !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const res = await fetch(normalizePath(path), options);
  const payload = await res.json();
  if (!res.ok) {
    const detail = payload?.detail?.error || payload?.error || {};
    throw new Error(`${detail.code || "ERROR"}: ${detail.message || JSON.stringify(payload)}`);
  }
  return payload;
}
