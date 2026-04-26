export function $(id) {
  return document.getElementById(id);
}

export function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text || "";
}

export function setMsg(id, text) {
  setText(id, text);
}
