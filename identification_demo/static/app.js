"use strict";

const fields = [
  ["id_number", "Số CCCD"], ["old_id_number", "Số CMND cũ"],
  ["name", "Họ và tên"], ["date_of_birth", "Ngày sinh"],
  ["gender", "Giới tính"], ["nationality", "Quốc tịch"],
  ["place_of_origin", "Quê quán"], ["place_of_residence", "Nơi thường trú"],
  ["issue_date", "Ngày cấp"], ["issuing_authority", "Cơ quan cấp"],
  ["identifying_features", "Đặc điểm nhân dạng"],
  ["expiry_date", "Ngày hết hạn"],
];
const sourceLabels = { qr_front: "QR · mặt trước", qr_back: "QR · mặt sau", ocr_front: "OCR · mặt trước", ocr_back: "OCR · mặt sau" };
const sides = ["front", "back"];
const $ = id => document.getElementById(id);
const selectedFiles = { front: null, back: null };
const previewURLs = { front: null, back: null };
let result = null;
let busy = false;
let toastTimer;
let serviceHealth = null;
let clientTimings = null;
let qualityTouched = false;

function selectedQuality() {
  return document.querySelector('input[name="quality"]:checked').value;
}

function updateGPUStatus() {
  const high = selectedQuality() === "high";
  const available = serviceHealth?.ocr_profiles?.high?.available;
  $("gpu-status").hidden = !high || available === true;
  $("gpu-status").textContent = !serviceHealth ? "Đang kiểm tra dịch vụ GPU…" : "Mức cao hiện chưa sẵn sàng. Thông tin QR vẫn được giữ; phần OCR cần dịch vụ GPU hoạt động.";
}

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
      text.textContent = values[key] || (key === "expiry_date" && result?.expiry?.expiry_type === "unlimited" ? "Không thời hạn" : key === "identifying_features" ? "Đã bỏ qua" : key === "old_id_number" ? "Không có / chưa đọc được" : "Chưa đọc được");
      content.append(text);
      if (!values[key]) value.classList.add("empty");
      if (values[key] && sourceLabels[result?.field_sources?.[key]]) {
        const source = document.createElement("span");
        source.className = "field-source";
        source.textContent = sourceLabels[result.field_sources[key]].replace("OCR", result.quality === "high" ? "OCR GPU" : "OCR CPU");
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
  clientTimings = null;
  renderFields();
  $("result-count").textContent = `0 / ${fields.length} trường`;
  $("json-output").textContent = "Chưa có kết quả.";
  $("result-notice").hidden = true;
  $("back-ocr-details").hidden = true;
  $("back-ocr-details").open = false;
  $("back-ocr-initial").replaceChildren();
  $("back-ocr-final").replaceChildren();
  $("back-ocr-candidate").textContent = "";
  $("back-ocr-authority").textContent = "";
  $("back-ocr-crop").hidden = true;
  $("back-ocr-crop-image").removeAttribute("src");
  $("timing-details").hidden = true;
  $("timing-details").open = false;
  $("copy-json").disabled = true;
  $("download-json").disabled = true;
  $("processing-time").textContent = "Ưu tiên QR · OCR bổ sung";
  $("result-sources").textContent = "";
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
  $("quality-selector").disabled = value;
  for (const side of sides) {
    for (const suffix of ["remove-file", "file-input", "choose-file"]) $(`${side}-${suffix}`).disabled = value;
  }
}

async function extract() {
  if (busy || sides.some(side => !selectedFiles[side])) return;
  const requestClock = { startedAt: performance.now() };
  uploadError("");
  resetResult();
  setBusy(true);
  setState("processing", "Đang gửi hai ảnh", "Thời gian chờ gồm tải ảnh lên, xử lý và nhận kết quả.");
  $("extract-label").textContent = "Đang gửi ảnh…";
  const form = new FormData();
  form.append("quality", selectedQuality());
  form.append("include_back_ocr", "true");
  for (const side of sides) form.append(`${side}_image`, selectedFiles[side]);
  try {
    const response = await sendCard(form, requestClock);
    let payload;
    try { payload = JSON.parse(response.responseText); }
    catch { throw new Error("Dịch vụ chưa trả về kết quả hợp lệ. Hãy thử lại sau."); }
    requestClock.payloadReadyAt = performance.now();
    if (response.status < 200 || response.status >= 300 || !payload.success) throw new Error(typeof payload.error === "string" ? payload.error : "Không trích xuất được ảnh. Hãy thử lại.");
    if (!payload.fields || typeof payload.fields !== "object" || Array.isArray(payload.fields)) throw new Error("Kết quả không đúng định dạng.");
    for (const [key] of fields) if (typeof payload.fields[key] !== "string") throw new Error("Kết quả thiếu trường thông tin.");
    result = payload;
    renderFields(result.fields);
    renderBackOCR(result);
    const count = fields.filter(([key]) => result.fields[key]).length;
    const qrCount = fields.filter(([key]) => result.field_sources?.[key]?.startsWith("qr_")).length;
    const ocrCount = count - qrCount;
    $("result-count").textContent = `${count} / ${fields.length} trường`;
    $("json-output").textContent = JSON.stringify(result, null, 2);
    $("copy-json").disabled = false;
    $("download-json").disabled = false;
    setState(count ? "success" : "", result.status === "complete" ? "Đã đọc xong thông tin CCCD" : count ? "Đã đọc một phần thông tin" : "Chưa tìm thấy thông tin", count ? `Giữ ${qrCount} trường từ QR và bổ sung ${ocrCount} trường bằng OCR.` : "Hãy dùng ảnh chụp thẳng, rõ chữ và không chói sáng.");
    const messages = Array.isArray(result.warnings) ? [...result.warnings] : [];
    if (result.status !== "complete") messages.push("Một số thông tin chưa đầy đủ. Hãy đối chiếu với thẻ gốc.");
    $("result-notice").textContent = messages.join(" ");
    $("result-notice").hidden = !messages.length;
    clientTimings = browserTimings(requestClock, result.processing_time_ms);
    renderProcessingSummary(result, qrCount, ocrCount);
    renderTimings(result);
    await afterResultPaint();
    clientTimings = browserTimings(requestClock, result.processing_time_ms);
    renderProcessingSummary(result, qrCount, ocrCount);
    renderTimings(result);
  } catch (error) {
    uploadError(error instanceof TypeError ? "Mất kết nối với dịch vụ. Kiểm tra kết nối và thử lại." : error.message);
    setState("", "Chưa thể đọc ảnh", "Hai ảnh đã chọn vẫn được giữ để bạn có thể thử lại.");
  } finally {
    setBusy(false);
  }
}

function renderBackOCR(payload) {
  const debug = payload.ocr?.back?.debug;
  if (!debug) return;
  const renderLines = (id, lines) => {
    const list = $(id);
    for (const line of Array.isArray(lines) ? lines : []) {
      const item = document.createElement("li");
      item.textContent = line.text || "(Không đọc được chữ)";
      if (line.in_authority) {
        item.classList.add("authority-line");
        item.title = "Được chọn để ghép cơ quan cấp";
      }
      list.append(item);
    }
    if (!list.children.length) {
      const item = document.createElement("li");
      item.textContent = "Không có dòng chữ nào được nhận dạng.";
      list.append(item);
    }
  };
  renderLines("back-ocr-initial", debug.initial_lines);
  renderLines("back-ocr-final", debug.final_lines);
  $("back-ocr-candidate").textContent = debug.ambiguous ? "Chưa chọn được khối cơ quan cấp" : debug.authority_candidate || "Chưa chọn được dòng phù hợp";
  $("back-ocr-authority").textContent = payload.fields.issuing_authority || "Chưa đọc được";
  const crop = debug.search_crop;
  if (crop?.image?.startsWith("data:image/jpeg;base64,") && Array.isArray(crop.bbox) && crop.bbox.length === 4) {
    $("back-ocr-crop-image").src = crop.image;
    $("back-ocr-crop-caption").textContent = `Vùng ảnh đã tìm bổ sung cơ quan cấp · ${Math.round(crop.bbox[2] - crop.bbox[0])} × ${Math.round(crop.bbox[3] - crop.bbox[1])} px`;
    $("back-ocr-crop").hidden = false;
  }
  $("back-ocr-details").hidden = false;
  $("back-ocr-details").open = true;
}

function sendCard(form, clock) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/cccd/extract");
    request.upload.onprogress = event => {
      if (clock.uploadFinishedAt) return;
      const sentMB = (event.loaded / 1024 / 1024).toLocaleString("vi-VN", { maximumFractionDigits: 1 });
      const progress = event.lengthComputable ? ` (${Math.round(event.loaded / event.total * 100)}%)` : "";
      $("extract-label").textContent = `Đang gửi ảnh${progress}…`;
      setState("processing", "Đang gửi hai ảnh", `Đã gửi ${sentMB} MB${progress}. Ảnh lớn hoặc đường truyền chậm sẽ cần thêm thời gian.`);
    };
    request.upload.onload = () => {
      clock.uploadFinishedAt = performance.now();
      $("extract-label").textContent = "Đang chờ kết quả…";
      setState("processing", "Đã gửi ảnh, đang chờ kết quả", "Dịch vụ đang đọc QR và bổ sung thông tin bằng OCR.");
    };
    request.onreadystatechange = () => {
      if (request.readyState === XMLHttpRequest.HEADERS_RECEIVED) {
        clock.headersAt = performance.now();
        $("extract-label").textContent = "Đang nhận kết quả…";
      }
    };
    request.onload = () => {
      clock.receivedAt = performance.now();
      resolve(request);
    };
    request.onerror = () => reject(new TypeError("Network error"));
    request.onabort = () => reject(new Error("Yêu cầu đã bị hủy. Hãy thử lại."));
    request.ontimeout = () => reject(new Error("Đã hết thời gian chờ phản hồi. Hãy thử lại."));
    request.send(form);
  });
}

function browserTimings(clock, pipelineMs) {
  const finished = performance.now();
  const headers = clock.headersAt ?? clock.receivedAt;
  const uploaded = Math.min(clock.uploadFinishedAt ?? headers, headers);
  const elapsed = (end, start) => Math.max(0, end - start);
  return {
    send_upload: elapsed(uploaded, clock.startedAt),
    wait_response: elapsed(headers, uploaded),
    receive_parse: elapsed(clock.payloadReadyAt, headers),
    render: elapsed(finished, clock.payloadReadyAt),
    total: elapsed(finished, clock.startedAt),
    outside_pipeline: Math.max(0, finished - clock.startedAt - pipelineMs),
  };
}

function afterResultPaint() {
  return new Promise(resolve => {
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      clearTimeout(deadline);
      resolve();
    };
    const deadline = setTimeout(finish, 150);
    if (document.hidden) finish();
    else requestAnimationFrame(() => setTimeout(finish, 0));
  });
}

function renderProcessingSummary(payload, qrCount, ocrCount) {
  const seconds = value => (value / 1000).toLocaleString("vi-VN", { maximumFractionDigits: 1 });
  $("processing-time").textContent = `Tổng chờ: ${seconds(clientTimings.total)} giây · QR/OCR: ${seconds(payload.processing_time_ms)} giây`;
  $("result-sources").textContent = `${qrCount} trường từ QR · ${ocrCount} trường từ OCR ${payload.quality === "high" ? "GPU" : "CPU"}`;
}

function renderTimings(payload) {
  const rows = $("timing-rows");
  rows.replaceChildren();
  const add = (label, value, detail = false) => {
    if (!Number.isFinite(value)) return;
    const row = document.createElement("tr");
    if (detail) row.className = "timing-detail";
    const name = document.createElement("td");
    const duration = document.createElement("td");
    name.textContent = label;
    duration.textContent = `${value.toLocaleString("vi-VN", { maximumFractionDigits: 1 })} ms`;
    row.append(name, duration);
    rows.append(row);
  };
  if (clientTimings) {
    add("Tổng chờ từ lúc bấm đến khi hiện kết quả", clientTimings.total);
    add("Chuẩn bị kết nối và gửi ảnh", clientTimings.send_upload, true);
    add("Chờ phản hồi sau khi gửi ảnh", clientTimings.wait_response, true);
    add("Nhận và đọc kết quả", clientTimings.receive_parse, true);
    add("Hiển thị kết quả", clientTimings.render, true);
    add("Thời gian ngoài pipeline QR/OCR", clientTimings.outside_pipeline);
  }
  const httpLabels = { server_until_result: "Server: từ nhận request đến có kết quả",
    receive_upload: "Server: nhận ảnh / đọc multipart", prepare_request: "Server: chuẩn bị request",
    dispatch_wait: "Server: chờ luồng xử lý HTTP", finish_result: "Server: hoàn tất kết quả" };
  for (const [key, label] of Object.entries(httpLabels)) add(label, payload.request_timings_ms?.[key], key !== "server_until_result");
  const labels = { image_decode: "Chuẩn bị ảnh", qr_front: "QR mặt trước", qr_back: "QR mặt sau",
    ocr_front: "OCR mặt trước", ocr_back: "OCR mặt sau", merge: "Tổng hợp kết quả", total: "Pipeline QR/OCR trên server" };
  const details = { queue_wait: "Chờ hàng đợi", inference: "Xử lý OCR", model_init: "Khởi tạo model",
    preprocess: "Tiền xử lý", card_preprocessing: "Hiệu chỉnh ảnh", paddle_detect_recognize: "PaddleOCR", vietocr_reread: "VietOCR đọc bổ sung",
    paddle_detection: "Tìm dòng chữ (Paddle)", line_crops: "Cắt dòng chữ", docu_recognition: "Nhận dạng chữ (DocU)",
    text_enhancement: "Làm rõ vùng chữ", enhanced_recognition: "Đọc lại vùng đã làm rõ",
    iqa: "Kiểm tra chất lượng ảnh", segmentation: "Tìm vùng thẻ", rotation: "Chỉnh hướng thẻ",
    detection: "Tìm vùng chữ", recognition: "Nhận dạng chữ", authority_refinement: "Đọc lại cơ quan cấp",
    metadata: "Trích xuất thông tin", recovery: "Khôi phục ảnh khó" };
  for (const [key, label] of Object.entries(labels)) {
    add(label, payload.timings_ms?.[key]);
    const side = key === "ocr_front" ? "front" : key === "ocr_back" ? "back" : null;
    if (side) for (const [stage, name] of Object.entries(details)) add(name, payload.ocr?.[side]?.timings_ms?.[stage], true);
  }
  $("ocr-work").textContent = sides.map(side => {
    const work = payload.ocr?.[side]?.work;
    const prep = payload.ocr?.[side]?.image_preprocessing;
    const adjustments = [["brightness", "tăng sáng"], ["contrast", "tăng tương phản"], ["denoise", "giảm nhiễu"]]
      .filter(([key]) => prep?.enhancement?.[key]).map(([, label]) => label);
    const notes = adjustments;
    return work ? `${side === "front" ? "Mặt trước" : "Mặt sau"}: ${work.primary_regions} vùng chữ, đọc bổ sung ${work.reread_regions} vùng.${notes.length ? ` ${notes.join(", ")}.` : ""}` : "";
  }).filter(Boolean).join(" ");
  $("timing-details").hidden = !rows.children.length;
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
$("quality-selector").addEventListener("change", () => {
  qualityTouched = true;
  uploadError("");
  resetResult();
  updateGPUStatus();
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
window.addEventListener("dragover", event => event.preventDefault());
window.addEventListener("drop", event => event.preventDefault());
$("copy-json").addEventListener("click", () => { if (result) copyText(JSON.stringify(result, null, 2)); });
$("copy-back-ocr").addEventListener("click", () => {
  const debug = result?.ocr?.back?.debug;
  if (!debug) return;
  const lines = rows => (Array.isArray(rows) ? rows : []).map(row => `${row.number}. ${row.text}`).join("\n");
  copyText(`OCR lần đầu:\n${lines(debug.initial_lines)}\n\nSau đọc bổ sung:\n${lines(debug.final_lines)}\n\nChữ ghép thành cơ quan cấp: ${debug.authority_candidate || ""}\nGiá trị trả về: ${result.fields.issuing_authority}`);
});
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
  serviceHealth = health;
  if (!qualityTouched && !busy && ["medium", "high"].includes(health.default_quality)) {
    document.querySelector(`input[name="quality"][value="${health.default_quality}"]`).checked = true;
  }
  updateGPUStatus();
  const available = health.qr?.available && (selectedQuality() === "high"
    ? health.ocr_profiles?.high?.available : health.engine?.source_available);
  $("service-state").className = `service-state ${available ? "ready" : "unavailable"}`;
  const dot = document.createElement("span");
  dot.className = "status-dot";
  $("service-state").replaceChildren(dot, document.createTextNode(available ? "Dịch vụ đang hoạt động" : "Một bộ đọc chưa sẵn sàng"));
}).catch(() => {
  $("service-state").textContent = "Chưa kết nối dịch vụ";
  $("service-state").className = "service-state unavailable";
});
