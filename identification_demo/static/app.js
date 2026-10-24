"use strict";

const fields = [
  ["id_number", "Số CCCD"], ["old_id_number", "Số CMND cũ"],
  ["name", "Họ và tên"], ["date_of_birth", "Ngày sinh"],
  ["gender", "Giới tính"], ["nationality", "Quốc tịch"],
  ["place_of_origin", "Quê quán"], ["place_of_residence", "Nơi thường trú"],
  ["issue_date", "Ngày cấp"], ["issuing_authority", "Cơ quan cấp"],
  ["identifying_features", "Đặc điểm nhân dạng"],
];
const sourceLabels = { qr_front: "QR · mặt trước", qr_back: "QR · mặt sau", ocr_front: "OCR · mặt trước", ocr_back: "OCR · mặt sau" };
const sides = ["front", "back"];
const $ = id => document.getElementById(id);
const selectedFiles = { front: null, back: null };
const previewURLs = { front: null, back: null };
let result = null;
let busy = false;
let toastTimer;

function makeIcon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("icon");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

function renderFields(values = null) {
  $("field-list").replaceChildren();
  for (const [key, label] of fields) {
    const row = document.createElement("div");
    row.className = "field-row";
    const name = document.createElement("dt");
    name.textContent = label;
    const value = document.createElement("dd");
    if (values === null) {
      const placeholder = document.createElement("span");
      placeholder.className = "placeholder-line";
      placeholder.setAttribute("aria-hidden", "true");
      value.append(placeholder);
      value.setAttribute("aria-label", "Chưa có kết quả");
    } else {
      const content = document.createElement("span");
      content.className = "field-value";
      const text = document.createElement("span");
      text.textContent = values[key] || (key === "old_id_number" ? "Không có / chưa đọc được" : "Chưa đọc được");
      content.append(text);
      if (!values[key]) value.classList.add("empty");
      if (values[key] && sourceLabels[result?.field_sources?.[key]]) {
        const source = document.createElement("span");
        source.className = "field-source";
        source.textContent = sourceLabels[result.field_sources[key]];
        content.append(source);
      }
      value.append(content);
    }
    if (values?.[key]) {
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "icon-button";
      copy.setAttribute("aria-label", `Sao chép ${label.toLowerCase()}`);
      copy.title = `Sao chép ${label.toLowerCase()}`;
      copy.append(makeIcon("copy"));
      copy.addEventListener("click", () => copyText(values[key]));
      value.append(copy);
    }
    row.append(name, value);
    $("field-list").append(row);
  }
}

function setState(kind, title, description) {
  $("result-state").className = `result-state ${kind}`;
  $("result-state-title").textContent = title;
  $("result-state-description").textContent = description;
}

function resetResult() {
  result = null;
  renderFields();
  $("result-count").textContent = `0 / ${fields.length} trường`;
  $("json-output").textContent = "Chưa có kết quả.";
  $("result-notice").hidden = true;
  $("copy-json").disabled = true;
  $("download-json").disabled = true;
  $("processing-time").textContent = "Ưu tiên QR · OCR bổ sung";
  setTab("fields");
  setState("", "Sẵn sàng đọc ảnh", "Tải đủ hai mặt để đọc QR và bổ sung thông tin bằng OCR.");
}

function uploadError(message) {
  $("upload-error").textContent = message;
  $("upload-error").hidden = !message;
}

function updateButton() {
  $("extract-button").disabled = busy || sides.some(side => !selectedFiles[side]);
}

function chooseFile(side, file) {
  if (busy || !file) return;
  $(`${side}-file-input`).value = "";
  if (!file.size) return uploadError("Ảnh đang trống. Hãy chọn lại ảnh CCCD.");
  if (file.size > 10 * 1024 * 1024) return uploadError("Mỗi ảnh tối đa 10 MB. Hãy chọn ảnh có dung lượng nhỏ hơn.");
  if (!["image/jpeg", "image/png"].includes(file.type)) return uploadError("Chỉ hỗ trợ ảnh JPG hoặc PNG.");
  if (previewURLs[side]) URL.revokeObjectURL(previewURLs[side]);
  selectedFiles[side] = file;
  previewURLs[side] = URL.createObjectURL(file);
  $(`${side}-preview-image`).src = previewURLs[side];
  $(`${side}-file-name`).textContent = file.name;
  $(`${side}-file-size`).textContent = file.size >= 1024 * 1024 ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(file.size / 1024))} KB`;
  $(`${side}-dropzone`).hidden = true;
  $(`${side}-preview`).hidden = false;
  uploadError("");
  resetResult();
  updateButton();
}

function clearSelection(side) {
  if (busy) return;
  selectedFiles[side] = null;
  if (previewURLs[side]) URL.revokeObjectURL(previewURLs[side]);
  previewURLs[side] = null;
  $(`${side}-preview-image`).removeAttribute("src");
  $(`${side}-file-input`).value = "";
  $(`${side}-dropzone`).hidden = false;
  $(`${side}-preview`).hidden = true;
  uploadError("");
  resetResult();
  updateButton();
}

function setBusy(value) {
  busy = value;
  updateButton();
  $("result-panel").setAttribute("aria-busy", String(value));
  $("extract-button").classList.toggle("processing", value);
  $("extract-label").textContent = value ? "Đang đọc QR và OCR…" : "Trích xuất thông tin";
  $("extract-spinner").hidden = !value;
  $("extract-arrow").hidden = value;
  for (const side of sides) {
    for (const suffix of ["remove-file", "file-input", "choose-file"]) $(`${side}-${suffix}`).disabled = value;
  }
}

async function extract() {
  if (busy || sides.some(side => !selectedFiles[side])) return;
  uploadError("");
  resetResult();
  setBusy(true);
  setState("processing", "Đang đọc hai mặt CCCD", "Đọc QR trước, sau đó dùng OCR bổ sung. Quá trình này có thể mất một lúc; vui lòng giữ trang đang mở.");
  const form = new FormData();
  for (const side of sides) form.append(`${side}_image`, selectedFiles[side]);
  try {
    const response = await fetch("/api/cccd/extract", { method: "POST", body: form });
    let payload;
    try { payload = await response.json(); }
    catch { throw new Error("Dịch vụ chưa trả về kết quả hợp lệ. Hãy thử lại sau."); }
    if (!response.ok || !payload.success) throw new Error(typeof payload.error === "string" ? payload.error : "Không trích xuất được ảnh. Hãy thử lại.");
    if (!payload.fields || typeof payload.fields !== "object" || Array.isArray(payload.fields)) throw new Error("Kết quả không đúng định dạng.");
    for (const [key] of fields) if (typeof payload.fields[key] !== "string") throw new Error("Kết quả thiếu trường thông tin.");
    result = payload;
    renderFields(result.fields);
    const count = fields.filter(([key]) => result.fields[key]).length;
    const qrCount = fields.filter(([key]) => result.field_sources?.[key]?.startsWith("qr_")).length;
    const ocrCount = count - qrCount;
    $("result-count").textContent = `${count} / ${fields.length} trường`;
    $("json-output").textContent = JSON.stringify(result, null, 2);
    $("processing-time").textContent = `${(result.processing_time_ms / 1000).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} giây · ${qrCount} QR, ${ocrCount} OCR`;
    $("copy-json").disabled = false;
    $("download-json").disabled = false;
    setState(count ? "success" : "", result.status === "complete" ? "Đã đọc xong thông tin CCCD" : count ? "Đã đọc một phần thông tin" : "Chưa tìm thấy thông tin", count ? `Giữ ${qrCount} trường từ QR và bổ sung ${ocrCount} trường bằng OCR.` : "Hãy dùng ảnh chụp thẳng, rõ chữ và không chói sáng.");
    const messages = Array.isArray(result.warnings) ? [...result.warnings] : [];
    if (result.status !== "complete") messages.push("Một số trường còn thiếu. Hãy thử ảnh rõ hơn và đối chiếu với thẻ gốc.");
    $("result-notice").textContent = messages.join(" ");
    $("result-notice").hidden = !messages.length;
  } catch (error) {
    uploadError(error instanceof TypeError ? "Mất kết nối với dịch vụ. Kiểm tra kết nối và thử lại." : error.message);
    setState("", "Chưa thể đọc ảnh", "Hai ảnh đã chọn vẫn được giữ để bạn có thể thử lại.");
  } finally {
    setBusy(false);
  }
}

function setTab(tab, focus = false) {
  for (const name of ["fields", "json"]) {
    $("tab-" + name).setAttribute("aria-selected", String(name === tab));
    $("tab-" + name).tabIndex = name === tab ? 0 : -1;
    $(name + "-view").hidden = name !== tab;
  }
  if (focus) $("tab-" + tab).focus();
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 2600);
}

async function copyText(value) {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(value);
    else {
      // Clipboard API is unavailable on an ordinary HTTP demo on a remote host.
      const textarea = document.createElement("textarea");
      textarea.className = "visually-hidden";
      textarea.value = value;
      document.body.append(textarea);
      const previousFocus = document.activeElement;
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      previousFocus?.focus();
      if (!copied) throw new Error("Clipboard unavailable");
    }
    toast("Đã sao chép vào bộ nhớ tạm");
  } catch { toast("Không sao chép được. Bạn có thể tải file JSON."); }
}

for (const side of sides) {
  $(`${side}-choose-file`).addEventListener("click", () => $(`${side}-file-input`).click());
  $(`${side}-file-input`).addEventListener("change", event => chooseFile(side, event.target.files[0]));
  $(`${side}-remove-file`).addEventListener("click", () => clearSelection(side));
  $(`${side}-preview-image`).addEventListener("error", () => {
    clearSelection(side);
    uploadError("Không xem trước được ảnh. Hãy chọn một ảnh JPG hoặc PNG hợp lệ.");
  });
  for (const eventName of ["dragenter", "dragover", "dragleave", "drop"]) {
    $(`${side}-dropzone`).addEventListener(eventName, event => {
      event.preventDefault();
      if (busy) return;
      $(`${side}-dropzone`).classList.toggle("dragover", ["dragenter", "dragover"].includes(eventName));
      if (eventName === "drop") chooseFile(side, event.dataTransfer.files[0]);
    });
  }
}
$("extract-button").addEventListener("click", extract);
for (const name of ["fields", "json"]) {
  $("tab-" + name).addEventListener("click", () => setTab(name));
  $("tab-" + name).addEventListener("keydown", event => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      setTab(event.key === "Home" ? "fields" : event.key === "End" ? "json" : name === "fields" ? "json" : "fields", true);
    }
  });
}
window.addEventListener("dragover", event => event.preventDefault());
window.addEventListener("drop", event => event.preventDefault());
$("copy-json").addEventListener("click", () => { if (result) copyText(JSON.stringify(result, null, 2)); });
$("download-json").addEventListener("click", () => {
  if (!result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "cccd-result.json";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener("beforeunload", () => {
  for (const url of Object.values(previewURLs)) if (url) URL.revokeObjectURL(url);
});

renderFields();
fetch("/api/health").then(response => {
  if (!response.ok) throw new Error("Health check failed");
  return response.json();
}).then(health => {
  const available = health.engine?.source_available && health.qr?.available;
  $("service-state").className = `service-state ${available ? "ready" : "unavailable"}`;
  const dot = document.createElement("span");
  dot.className = "status-dot";
  $("service-state").replaceChildren(dot, document.createTextNode(available ? "Dịch vụ đang hoạt động" : "Một bộ đọc chưa sẵn sàng"));
}).catch(() => {
  $("service-state").textContent = "Chưa kết nối dịch vụ";
  $("service-state").className = "service-state unavailable";
});
