/* Separate state and API calls for the optional experiment workspace. */
(() => {
  "use strict";
  const byId = id => document.getElementById(id);
  const sideLabels = {front: "Mặt trước", back: "Mặt sau"};
  const labels = {
    id_number: "Số căn cước", old_id_number: "Số CMND cũ", name: "Họ và tên", date_of_birth: "Ngày sinh",
    gender: "Giới tính", nationality: "Quốc tịch", place_of_origin: "Quê quán", place_of_residence: "Nơi cư trú",
    issue_date: "Ngày cấp", issuing_authority: "Cơ quan cấp", identifying_features: "Đặc điểm nhân dạng", expiry_date: "Ngày hết hạn",
  };
  const checkLabels = {id_number: "Số căn cước 12 số", document_number: "Số tài liệu MRZ", date_of_birth: "Ngày sinh", expiry_date: "Ngày hết hạn", composite: "Checksum tổng hợp"};
  const statusLabels = {verified: "MRZ vượt qua các kiểm tra số và ngày", partial: "Một số trường MRZ chưa đủ tin cậy; chỉ dùng trường vượt kiểm tra", invalid: "MRZ sai cấu trúc hoặc checksum", not_found: "Chưa tìm được MRZ", error: "Chưa xử lý được MRZ", unsupported: "MRZ không thuộc mẫu CCCD Việt Nam được hỗ trợ"};
  let active = "extract", healthChecked = false, busy = false, currentResult = null;
  const urls = {};

  async function showWorkspace(name) {
    active = name;
    for (const workspace of ["extract", "experiment"]) {
      const selected = workspace === name;
      byId(`workspace-${workspace}`).hidden = !selected;
      const button = byId(`workspace-tab-${workspace}`);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    }
    if (name === "experiment" && !healthChecked) {
      healthChecked = true;
      try {
        const response = await fetch("/api/cccd/experiment/health");
        if (!response.ok) throw new Error("health");
        const info = await response.json();
        byId("lab-health").textContent = info.available ? "DocAligner và FastMRZ sẵn sàng · OCR dùng GPU hiện có." : (info.reason || "Worker thử nghiệm chưa sẵn sàng.");
      } catch (_) { byId("lab-health").textContent = "Chưa kết nối được bộ xử lý thử nghiệm."; }
    }
  }
  for (const name of ["extract", "experiment"]) {
    const button = byId(`workspace-tab-${name}`);
    button.addEventListener("click", () => showWorkspace(name));
    button.addEventListener("keydown", event => {
      if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const target = event.key === "Home" ? "extract" : event.key === "End" ? "experiment" : active === "extract" ? "experiment" : "extract";
        showWorkspace(target);
        byId(`workspace-tab-${target}`).focus();
      }
    });
  }

  function clearResult() {
    currentResult = null;
    byId("lab-result").hidden = true;
    byId("lab-images").replaceChildren();
    byId("lab-json").textContent = "";
    byId("lab-mrz-text").textContent = "";
    byId("lab-fields").replaceChildren();
    byId("lab-error").hidden = true;
  }
  function error(message) {
    byId("lab-error").textContent = message;
    byId("lab-error").hidden = false;
  }
  for (const side of ["front", "back"]) {
    byId(`lab-${side}`).addEventListener("change", () => {
      clearResult();
      if (urls[side]) URL.revokeObjectURL(urls[side]);
      const file = byId(`lab-${side}`).files[0];
      const image = byId(`lab-${side}-input-preview`);
      image.hidden = true;
      image.removeAttribute("src");
      if (!file) return;
      if (!["image/jpeg", "image/png"].includes(file.type) || !file.size || file.size > 10 * 1024 * 1024) {
        byId(`lab-${side}`).value = "";
        error("Chọn ảnh JPG/PNG có nội dung, tối đa 10 MB mỗi ảnh.");
        return;
      }
      urls[side] = URL.createObjectURL(file);
      image.src = urls[side]; image.hidden = false;
    });
    byId(`lab-${side}-rotation`).addEventListener("change", clearResult);
  }
  byId("lab-align").addEventListener("change", clearResult);

  function textElement(tag, text, className) {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    return element;
  }
  function sourceLabel(source) {
    if (!source) return "—";
    if (source.startsWith("mrz")) return "MRZ · mặt sau";
    const side = source.includes("front") ? "mặt trước" : "mặt sau";
    const variant = source.includes("aligned") ? "ảnh căn chỉnh" : "ảnh gốc";
    return `${source.startsWith("qr") ? "QR" : "OCR GPU"} · ${side} · ${variant}`;
  }
  function cornerOverlay(side, data) {
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `${sideLabels[side]} gốc và bốn góc tìm được`);
    const source = byId(`lab-${side}-input-preview`);
    const draw = () => {
      const [width, height] = data.input_size;
      const scale = Math.min(1, 900 / Math.max(width, height));
      canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
      const context = canvas.getContext("2d");
      context.scale(scale, scale);
      context.save();
      context.translate(width / 2, height / 2);
      context.rotate((data.rotation_clockwise || 0) * Math.PI / 180);
      context.drawImage(source, -source.naturalWidth / 2, -source.naturalHeight / 2);
      context.restore();
      const corners = data.alignment.corners;
      if (corners?.length === 4) {
        context.beginPath();
        corners.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
        context.closePath(); context.lineWidth = Math.max(3, width / 300); context.strokeStyle = "#03ac8b"; context.stroke();
        context.fillStyle = "#03ac8b";
        corners.forEach(([x, y]) => { context.beginPath(); context.arc(x, y, width / 120, 0, 2 * Math.PI); context.fill(); });
      }
    };
    if (source.complete && source.naturalWidth) draw();
    else source.addEventListener("load", draw, {once: true});
    return canvas;
  }
  function render(result) {
    currentResult = result;
    byId("lab-result").hidden = false;
    const review = result.review_fields || [];
    byId("lab-summary").textContent = `${result.recognized_fields} / ${result.total_fields} trường · ${(result.processing_time_ms / 1000).toFixed(1)} giây${result.status === "identity_conflict" ? " · Hai ảnh có số căn cước mâu thuẫn; chưa ghép dữ liệu." : review.length ? " · Có trường cần đối chiếu." : ""}`;
    byId("lab-warnings").replaceChildren(...result.warnings.map(message => textElement("li", message)));
    const rows = Object.entries(labels).map(([key, label]) => {
      const row = document.createElement("tr");
      if (review.includes(key)) row.className = "lab-review";
      const source = textElement("td", sourceLabel(result.field_sources[key]));
      const conflict = result.conflicts.find(item => item.field === key);
      if (conflict) source.append(textElement("div", "Đọc khác nhau: " + conflict.candidates.map(item => `${item.value} (${sourceLabel(item.source)})`).join("; "), "lab-conflict"));
      else if (review.includes(key)) source.append(textElement("div", "Cần đối chiếu thẻ gốc", "lab-conflict"));
      row.append(textElement("th", label), textElement("td", result.fields[key] || "Chưa đọc được"), source);
      row.firstChild.scope = "row";
      return row;
    });
    byId("lab-fields").replaceChildren(...rows);
    byId("lab-images").replaceChildren();
    for (const [side, data] of Object.entries(result.sides)) {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.alt = `${sideLabels[side]} dùng để đọc chữ`;
      if (data.preview?.startsWith("data:image/jpeg;base64,")) image.src = data.preview;
      const aligned = data.alignment.status === "aligned";
      figure.append(textElement("p", `${sideLabels[side]} · Ảnh gốc và bốn góc`), cornerOverlay(side, data),
        textElement("p", "Ảnh dùng để đọc chữ"), image,
        textElement("figcaption", `${sideLabels[side]} · ${aligned ? "Đã căn chỉnh" : "Giữ ảnh gốc"}${data.alignment.confidence != null ? ` · Điểm tìm thẻ ${data.alignment.confidence.toFixed(2)}` : ""}${data.ocr_image ? " · OCR dùng ảnh gốc dự phòng" : ""}`));
      byId("lab-images").append(figure);
    }
    const mrz = result.sides.back.mrz || {};
    byId("lab-mrz-status").textContent = statusLabels[mrz.status] || "Chưa có kết quả MRZ";
    const stopLabels = {usable_fields: "Đã có trường MRZ vượt kiểm tra", covered_by_qr_ocr: "Các nguồn khác đã cung cấp đủ thông tin", time_budget: "Dừng thử thêm để giảm thời gian chờ", attempt_limit: "Đã đủ số lượt thử"};
    byId("lab-mrz-attempts").textContent = `${mrz.attempts?.length || 0} lượt đọc${mrz.elapsed_ms != null ? ` · ${Math.round(mrz.elapsed_ms)} ms` : ""}${stopLabels[mrz.stop_reason] ? ` · ${stopLabels[mrz.stop_reason]}` : ""}`;
    byId("lab-mrz-text").textContent = mrz.raw_text || "Chưa đọc được dòng MRZ.";
    byId("lab-mrz-checks").replaceChildren(...Object.entries(mrz.checks || {}).map(([key, passed]) => textElement("span", `${passed ? "✓" : "×"} ${checkLabels[key] || key}`, passed ? "lab-check-pass" : "lab-check-fail")));
    // Keep large previews in download, omit them from the readable JSON panel.
    byId("lab-json").textContent = JSON.stringify(result, (key, value) => key === "preview" ? "[ảnh xem trước]" : value, 2);
  }

  byId("lab-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (busy) return;
    clearResult();
    const body = new FormData(byId("lab-form"));
    body.set("align", String(byId("lab-align").checked));
    busy = true;
    for (const element of byId("lab-form").elements) element.disabled = true;
    byId("lab-progress").hidden = false;
    byId("lab-progress").textContent = "Đang căn chỉnh, đọc MRZ và chữ in trên hai mặt…";
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 250000);
    try {
      const response = await fetch("/api/cccd/experiment/extract", {method: "POST", body, signal: controller.signal});
      const result = await response.json();
      if (!response.ok || !result.success) throw new Error(result.error || "Chưa chạy được thử nghiệm.");
      render(result);
    } catch (failure) {
      error(failure.name === "AbortError" ? "Đã hết thời gian chờ thử nghiệm. Hãy thử lại." : failure.message || "Chưa kết nối được bộ xử lý thử nghiệm.");
    } finally {
      clearTimeout(timer);
      busy = false;
      for (const element of byId("lab-form").elements) element.disabled = false;
      byId("lab-progress").hidden = true;
    }
  });
  byId("lab-download").addEventListener("click", () => {
    if (!currentResult) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(currentResult, null, 2)], {type: "application/json"}));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "cccd-thu-nghiem.json"; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  window.addEventListener("pagehide", () => Object.values(urls).forEach(url => URL.revokeObjectURL(url)));
})();
