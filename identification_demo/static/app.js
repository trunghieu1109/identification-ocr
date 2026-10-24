"use strict";

const fieldSets = { front: [
  ["id_number", "Số CCCD"], ["name", "Họ và tên"],
  ["date_of_birth", "Ngày sinh"], ["gender", "Giới tính"],
  ["nationality", "Quốc tịch"], ["place_of_origin", "Quê quán"],
  ["place_of_residence", "Nơi thường trú"],
], back: [
  ["issue_date", "Ngày cấp"], ["issuing_authority", "Cơ quan cấp"],
  ["identifying_features", "Đặc điểm nhân dạng"],
], qr: [
  ["id_number", "Số căn cước"], ["old_id_number", "Số CMND cũ"],
  ["name", "Họ và tên"], ["date_of_birth", "Ngày sinh"],
  ["gender", "Giới tính"], ["place_of_residence", "Nơi cư trú"],
  ["issue_date", "Ngày cấp"],
] };
let fields = fieldSets.front;
const $ = id => document.getElementById(id);
let selectedFile = null;
let previewURL = null;
let result = null;
let busy = false;
let readMode = "ocr";
let serviceHealth = null;
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
      value.textContent = values[key] || (readMode === "qr" && key === "old_id_number" && result?.status === "complete" ? "Không có trong mã QR" : "Chưa đọc được");
      if (!values[key]) value.classList.add("empty");
    }
    if (values && values[key]) {
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
  fields = readMode === "qr" ? fieldSets.qr : $("card-side").value === "back" ? fieldSets.back : fieldSets.front;
  renderFields();
  $("result-count").textContent = `0 / ${fields.length} trường`;
  $("side-badge").textContent = readMode === "qr" ? "Mã QR" : { auto: "Tự nhận diện", front: "Mặt trước", back: "Mặt sau" }[$("card-side").value];
  $("raw-text-label").textContent = "Xem văn bản OCR";
  $("raw-text-details").hidden = true;
  $("raw-text-details").open = false;
  $("raw-text").textContent = "";
  $("json-output").textContent = "Chưa có kết quả.";
  $("result-notice").hidden = true;
  $("copy-json").disabled = true;
  $("download-json").disabled = true;
  $("processing-time").textContent = `${fields.length} trường thông tin`;
  setTab("fields");
  setState("", "Sẵn sàng đọc ảnh", "Kết quả sẽ xuất hiện sau khi bạn tải ảnh và trích xuất.");
}

function uploadError(message) {
  $("upload-error").textContent = message;
  $("upload-error").hidden = !message;
}

function chooseFile(file) {
  if (busy || !file) return;
  $("file-input").value = "";
  if (!file.size) return uploadError("Ảnh đang trống. Hãy chọn lại ảnh CCCD.");
  if (file.size > 10 * 1024 * 1024) return uploadError("Mỗi ảnh tối đa 10 MB. Hãy chọn ảnh có dung lượng nhỏ hơn.");
  if (!["image/jpeg", "image/png"].includes(file.type)) return uploadError("Chỉ hỗ trợ ảnh JPG hoặc PNG.");
  clearSelection();
  selectedFile = file;
  previewURL = URL.createObjectURL(file);
  $("preview-image").src = previewURL;
  $("file-name").textContent = file.name;
  $("file-size").textContent = file.size >= 1024 * 1024 ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(file.size / 1024))} KB`;
  $("dropzone").hidden = true;
  $("preview").hidden = false;
  $("extract-button").disabled = false;
}

function clearSelection() {
  if (busy) return;
  selectedFile = null;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  $("preview-image").removeAttribute("src");
  $("file-input").value = "";
  $("dropzone").hidden = false;
  $("preview").hidden = true;
  $("extract-button").disabled = true;
  uploadError("");
  resetResult();
}

function setBusy(value) {
  busy = value;
  $("result-panel").setAttribute("aria-busy", String(value));
  $("extract-button").disabled = value || !selectedFile;
  $("extract-button").classList.toggle("processing", value);
  $("extract-label").textContent = value ? "Đang đọc thông tin…" : readMode === "qr" ? "Đọc mã QR" : "Trích xuất thông tin";
  $("extract-spinner").hidden = !value;
  $("extract-arrow").hidden = value;
  $("remove-file").disabled = value;
  $("file-input").disabled = value;
  $("card-side").disabled = value;
  $("read-mode").disabled = value;
}

async function extract() {
  if (!selectedFile || busy) return;
  uploadError("");
  resetResult();
  setBusy(true);
  setState("processing", readMode === "qr" ? "Đang đọc mã QR" : "Đang nhận dạng thông tin", readMode === "qr" ? "Đang tìm mã QR và đọc thông tin trong ảnh." : "Lần đầu có thể lâu hơn do tải mô hình. Vui lòng giữ trang đang mở.");
  const form = new FormData();
  form.append("file", selectedFile);
  if (readMode === "ocr") form.append("side", $("card-side").value);
  try {
    const response = await fetch(readMode === "qr" ? "/api/qr/cccd" : "/api/ocr/cccd", { method: "POST", body: form });
    let payload;
    try { payload = await response.json(); }
    catch { throw new Error("Dịch vụ chưa trả về kết quả hợp lệ. Hãy thử lại sau."); }
    if (!response.ok || !payload.success) {
      const detail = typeof payload.error === "string" ? payload.error : "Không trích xuất được ảnh. Hãy thử lại.";
      throw new Error(detail);
    }
    if (!payload.fields || typeof payload.fields !== "object" || Array.isArray(payload.fields)) throw new Error("Kết quả không đúng định dạng.");
    const responseSide = payload.side === "back" ? "back" : "front";
    const responseFields = readMode === "qr" ? fieldSets.qr : fieldSets[responseSide];
    for (const [key] of responseFields) if (typeof payload.fields[key] !== "string") throw new Error("Kết quả thiếu trường thông tin.");
    fields = responseFields;
    result = payload;
    renderFields(result.fields);
    $("side-badge").textContent = readMode === "qr" ? "Mã QR" : responseSide === "back" ? "Mặt sau" : "Mặt trước";
    const count = fields.filter(([key]) => result.fields[key]).length;
    $("result-count").textContent = `${count} / ${fields.length} trường`;
    $("json-output").textContent = JSON.stringify(result, null, 2);
    $("processing-time").textContent = `Xử lý trong ${(result.processing_time_ms / 1000).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} giây`;
    $("copy-json").disabled = false;
    $("download-json").disabled = false;
    $("raw-text").textContent = readMode === "ocr" && typeof result.raw_text === "string" ? result.raw_text : "";
    $("raw-text-details").hidden = !$("raw-text").textContent;
    if (readMode === "qr") {
      const titles = { complete: "Đã đọc xong mã QR", partial: "Đã đọc một phần thông tin QR", no_qr: "Chưa đọc được mã QR", unsupported_qr: "Đã đọc QR, chưa tách được thông tin", multiple_qr: "Ảnh có nhiều mã QR" };
      setState(count ? "success" : "", titles[result.status] || "Đã xử lý ảnh", result.message || "Hãy đối chiếu kết quả với thẻ gốc.");
      $("result-notice").textContent = result.message || "";
      $("result-notice").hidden = result.status === "complete" || !result.message;
      return;
    }
    setState(count ? "success" : "", count ? "Đã đọc xong ảnh CCCD" : result.raw_text ? "Đã đọc văn bản, chưa tách được thông tin" : "Chưa tìm thấy thông tin", count ? `Trích xuất được ${count} trên ${fields.length} trường của ${responseSide === "back" ? "mặt sau" : "mặt trước"}.` : "Hãy chọn đúng mặt thẻ, dùng ảnh chụp thẳng và rõ chữ.");
    if (count < fields.length) {
      $("result-notice").textContent = count ? "Một số trường chưa đọc được. Bạn có thể xem văn bản OCR hoặc thử ảnh rõ hơn." : "Chưa tách được trường nào. Kiểm tra mặt thẻ đã chọn và mở văn bản OCR để xem kết quả nhận dạng.";
      $("result-notice").hidden = false;
    }
  } catch (error) {
    uploadError(error instanceof TypeError ? "Mất kết nối với dịch vụ. Kiểm tra kết nối và thử lại." : error.message);
    setState("", "Chưa thể đọc ảnh", "Ảnh đã chọn vẫn được giữ để bạn có thể thử lại.");
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

$("choose-file").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", event => chooseFile(event.target.files[0]));
$("remove-file").addEventListener("click", clearSelection);
$("extract-button").addEventListener("click", extract);
$("card-side").addEventListener("change", () => { uploadError(""); resetResult(); });
$("read-mode").addEventListener("change", event => {
  if (busy) return;
  readMode = event.target.value;
  $("side-selector").hidden = readMode === "qr";
  $("mode-description").textContent = readMode === "qr" ? "Lấy thông tin trực tiếp từ mã QR. Tải ảnh mặt thẻ có QR hoặc ảnh cắt riêng vùng QR." : "Đọc thông tin trên mặt trước hoặc mặt sau CCCD.";
  $("image-guidance").textContent = readMode === "qr" ? "Chụp rõ toàn bộ mã QR và viền xung quanh, tránh lóa hoặc mờ. Có thể cắt riêng vùng QR để dễ đọc hơn." : "Chụp thẳng, đủ sáng và rõ chữ. Để thẻ chiếm phần lớn khung hình, tránh chói sáng hoặc che thông tin.";
  uploadError("");
  resetResult();
  setBusy(false);
  updateServiceState();
});
$("preview-image").addEventListener("error", () => {
  clearSelection();
  uploadError("Không xem trước được ảnh. Hãy chọn một ảnh JPG hoặc PNG hợp lệ.");
});
for (const name of ["fields", "json"]) {
  $("tab-" + name).addEventListener("click", () => setTab(name));
  $("tab-" + name).addEventListener("keydown", event => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      setTab(event.key === "Home" ? "fields" : event.key === "End" ? "json" : name === "fields" ? "json" : "fields", true);
    }
  });
}
for (const eventName of ["dragenter", "dragover"]) {
  $("dropzone").addEventListener(eventName, event => {
    event.preventDefault();
    $("dropzone").classList.add("dragover");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  $("dropzone").addEventListener(eventName, event => {
    event.preventDefault();
    $("dropzone").classList.remove("dragover");
    if (eventName === "drop") chooseFile(event.dataTransfer.files[0]);
  });
}
// Prevent dropping a file outside the dropzone from navigating away from the demo.
window.addEventListener("dragover", event => event.preventDefault());
window.addEventListener("drop", event => event.preventDefault());
$("copy-json").addEventListener("click", () => { if (result) copyText(JSON.stringify(result, null, 2)); });
$("download-json").addEventListener("click", () => {
  if (!result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = readMode === "qr" ? "cccd-qr-result.json" : "cccd-result.json";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener("beforeunload", () => { if (previewURL) URL.revokeObjectURL(previewURL); });

function updateServiceState() {
  if (!serviceHealth) return;
  const available = readMode === "qr" ? serviceHealth.qr?.available : serviceHealth.engine?.source_available;
  $("service-state").className = `service-state ${available ? "ready" : "unavailable"}`;
  $("service-state").replaceChildren();
  const dot = document.createElement("span");
  dot.className = "status-dot";
  $("service-state").append(dot, document.createTextNode(available ? "Dịch vụ đang hoạt động" : readMode === "qr" ? "Bộ đọc QR chưa được cài đặt" : "OCR chưa được cài đặt"));
}

renderFields();
fetch("/api/health").then(response => {
  if (!response.ok) throw new Error("Health check failed");
  return response.json();
}).then(health => {
  serviceHealth = health;
  updateServiceState();
}).catch(() => {
  $("service-state").textContent = "Chưa kết nối dịch vụ";
  $("service-state").className = "service-state unavailable";
});
