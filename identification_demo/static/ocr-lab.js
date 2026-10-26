"use strict";

(() => {
  const el = id => document.getElementById(id);
  const engines = [
    { id: "namlhn_cpu", label: "namlhn CPU", model: "Transformer" },
    { id: "namlhn_gpu", label: "namlhn GPU", model: "Seq2Seq" },
    { id: "docu_gpu", label: "DocU GPU", model: "Seq2Seq" },
    { id: "docu_gpu_regions", label: "DocU GPU · v2 theo vùng", model: "Seq2Seq" },
    { id: "docu_gpu_v3", label: "DocU GPU · v3 mở rộng box", model: "Seq2Seq" },
  ];
  const fields = [
    ["id_number", "Số căn cước"], ["old_id_number", "Số CMND cũ"], ["name", "Họ và tên"],
    ["date_of_birth", "Ngày sinh"], ["gender", "Giới tính"], ["nationality", "Quốc tịch"],
    ["place_of_origin", "Quê quán"], ["place_of_residence", "Nơi thường trú"],
    ["issue_date", "Ngày cấp"], ["issuing_authority", "Cơ quan cấp"],
    ["identifying_features", "Đặc điểm nhân dạng"], ["expiry_date", "Ngày hết hạn · mặt trước"],
  ];
  const sourceNames = { qr_front: "QR · mặt trước", qr_back: "QR · mặt sau", ocr_front: "OCR · mặt trước", ocr_back: "OCR · mặt sau" };
  const sideNames = { front: "Mặt trước", back: "Mặt sau" };
  const statusNames = { complete: "Đọc đủ thông tin", partial: "Đọc được một phần", no_text: "Chưa đọc được thông tin" };
  const files = { front: null, back: null }, urls = { front: null, back: null };
  const reports = {}, elapsed = {};
  let running = false;
  let active = null;
  const seconds = ms => `${(Number(ms || 0) / 1000).toFixed(2)} s`;

  function error(message) {
    el("ocr-lab-error").textContent = message;
    el("ocr-lab-error").hidden = !message;
  }
  function controls() {
    const ready = files.front && files.back;
    for (const id of ["ocr-lab-run", "ocr-lab-run-all", "ocr-lab-run-docu"]) el(id).disabled = running || !ready;
    el("ocr-lab-run").querySelector(".spinner").hidden = !running;
    for (const side of ["front", "back"]) {
      for (const action of ["input", "choose", "remove"]) el(`ocr-lab-${side}-${action}`).disabled = running;
    }
    el("ocr-lab-engines").disabled = running;
    el("ocr-lab-use-qr").disabled = running;
    el("ocr-lab-results").setAttribute("aria-busy", String(running));
    for (const id of ["ocr-lab-copy", "ocr-lab-download"]) el(id).disabled = running || !Object.keys(reports).length;
  }
  function displayValue(report, key) {
    if (!report?.success) return "";
    return key === "expiry_date" && report.expiry.expiry_type === "unlimited" ? "Không thời hạn" : report.fields[key] || "";
  }
  function render() {
    for (const engine of engines) {
      const report = reports[engine.id];
      const status = el(`ocr-lab-summary-status-${engine.id}`);
      const info = el(`ocr-lab-summary-info-${engine.id}`);
      const warnings = el(`ocr-lab-summary-warnings-${engine.id}`);
      status.textContent = active === engine.id ? "Đang xử lý…" : report ? report.success ? statusNames[report.status] || report.status : "Lỗi xử lý" : "Chưa chạy";
      info.textContent = report?.success ? `${report.recognized_fields}/${report.total_fields} trường · Server ${seconds(report.processing_time_ms)} · Tổng chờ ${seconds(elapsed[engine.id])}` : report?.error || "Dùng cùng hai ảnh để so sánh.";
      warnings.textContent = report?.warnings?.join(" ") || "";
      if (report?.success && report.ocr_engine.used) {
        el(`ocr-lab-health-${engine.id}`).textContent = `${report.ocr_engine.model || engine.model}${report.ocr_engine.strategy === "namlhn_docu_v1" ? " · bộ đọc DocU" : report.ocr_engine.strategy === "docu_regions_v2" ? " · v2" : report.ocr_engine.strategy === "docu_expanded_v3" ? " · v3" : ""}${report.ocr_engine.decoder === "cuda_batched_v1" ? " · batch GPU" : ""} · Đã nạp model`;
      }
      status.className = `qr-method-status${report?.success && report.status === "complete" ? " success" : report ? " attention" : ""}`;
    }
    el("ocr-lab-comparison").replaceChildren();
    for (const [key, label] of fields) {
      const row = document.createElement("tr"), heading = document.createElement("th");
      heading.scope = "row";
      heading.textContent = label;
      row.append(heading);
      const successful = engines.filter(engine => reports[engine.id]?.success);
      const differs = new Set(successful.map(engine => displayValue(reports[engine.id], key))).size > 1;
      for (const engine of engines) {
        const cell = document.createElement("td"), report = reports[engine.id];
        const value = document.createElement("span"), source = document.createElement("small");
        const text = displayValue(report, key);
        value.textContent = text || (!report ? "Chưa chạy" : !report.success ? "Lỗi xử lý" : "Chưa đọc được");
        source.textContent = text ? sourceNames[report.field_sources[key]] || (report.expiry.expiry_type === "unlimited" && key === "expiry_date" ? "OCR · mặt trước" : "") : "";
        cell.append(value, source);
        if (differs && report?.success) cell.className = "ocr-value-differs";
        cell.dataset.engine = engine.id;
        cell.dataset.field = key;
        row.append(cell);
      }
      el("ocr-lab-comparison").append(row);
    }
    el("ocr-lab-details").replaceChildren();
    if (!Object.keys(reports).length) {
      const note = document.createElement("p");
      note.className = "mode-description";
      note.textContent = "Chi tiết sẽ xuất hiện sau khi chạy từng bộ xử lý.";
      el("ocr-lab-details").append(note);
    }
    for (const engine of engines) {
      const report = reports[engine.id];
      if (!report) continue;
      const details = document.createElement("details"), title = document.createElement("summary");
      details.className = "back-ocr-details";
      details.id = `ocr-lab-detail-${engine.id}`;
      title.textContent = `${engine.label} · ${report.ocr_engine?.model || engine.model}`;
      details.append(title);
      if (report.success) {
        const timing = document.createElement("p");
        timing.textContent = `QR: ${seconds(report.timings_ms.qr_front + report.timings_ms.qr_back)} · OCR mặt trước: ${seconds(report.timings_ms.ocr_front)} · OCR mặt sau: ${seconds(report.timings_ms.ocr_back)}. Thời gian server gồm chờ model; tổng chờ gồm gửi ảnh và nhận kết quả.`;
        details.append(timing);
        const columns = document.createElement("div");
        columns.className = "back-ocr-columns";
        for (const side of ["front", "back"]) {
          const section = document.createElement("section"), label = document.createElement("h3"), list = document.createElement("ol");
          label.textContent = sideNames[side];
          section.append(label);
          const recovery = report.ocr[side].field_recovery;
          if (recovery) {
            const note = document.createElement("p");
            const names = { recovered: "Đã lấy được ngày", not_found: "Chưa lấy được ngày", skipped: "Không cần hoặc chưa xác định được vùng đọc", error: "Chưa đọc lại được vùng ngày" };
            note.textContent = `${names[recovery.status] || recovery.status} · ${recovery.reread_regions || 0} vùng · ${Number(recovery.elapsed_ms || 0).toFixed(0)} ms bổ sung.`;
            section.append(note);
            const reasons = {
              already_found: "Lượt đầu đã có ngày hợp lệ.", not_requested: "Trường này đã có từ QR hoặc không được yêu cầu OCR.",
              label_not_found: "Chưa tìm được nhãn hoặc ngày đứng riêng để chọn vùng đọc.",
              ambiguous_label: "Có nhiều nhãn ở các vùng khác nhau.", ambiguous_date: "Có nhiều ngày chưa xác định được ý nghĩa.",
              no_valid_date: "Đã đọc vùng ảnh nhưng chưa ghép được nhãn với ngày hợp lệ.",
              conflict: "Các dòng chữ cho nhiều ngày khác nhau.", local_read_failed: "Lượt đọc vùng bị lỗi; giữ chữ ban đầu.",
              label_and_date: "Đã ghép được nhãn và ngày từ chữ trên ảnh.", invalid_region: "Vùng ảnh không hợp lệ.",
            };
            const reason = document.createElement("p");
            reason.textContent = reasons[recovery.reason] || recovery.reason || "";
            section.append(reason);
          }
          if (report.ocr[side].work) {
            const work = document.createElement("p");
            work.textContent = `${report.ocr[side].work.full_image_attempts || 0} lượt toàn ảnh · ${report.ocr[side].work.reread_regions || 0} vùng đọc lại.`;
            section.append(work);
          }
          const prep = report.ocr[side].image_preprocessing;
          if (prep) {
            const note = document.createElement("p");
            const adjustments = [["brightness", "Tăng sáng"], ["contrast", "Tăng tương phản"], ["denoise", "Giảm nhiễu"]]
              .filter(([key]) => prep.enhancement?.[key]).map(([, label]) => label);
            note.textContent = [prep.crop?.status === "cropped" ? "Đã crop vùng thẻ" : "Đọc toàn ảnh", ...adjustments].join(" · ");
            section.append(note);
          }
          for (const row of report.ocr[side].debug?.text_lines || []) {
            const item = document.createElement("li");
            item.textContent = row.text;
            item.value = row.number;
            if (row.is_expiry) item.className = "authority-line";
            list.append(item);
          }
          section.append(list);
          for (const read of report.ocr[side].debug?.region_reads || []) {
            const region = document.createElement("details"), summary = document.createElement("summary");
            summary.textContent = `${read.processing === 'v2_expanded_box_v1' ? 'Mở rộng box' : 'Vùng đọc lại'} · ${read.target === "issuing_authority" ? "Cơ quan cấp" : read.target === "expiry_date" ? "Ngày hết hạn" : "Ngày cấp"}`;
            region.append(summary);
            if (read.search_crop?.image?.startsWith("data:image/jpeg;base64,")) {
              const figure = document.createElement("figure"), preview = document.createElement("img");
              figure.className = "back-ocr-crop";
              preview.src = read.search_crop.image;
              preview.alt = "Vùng ảnh thực tế được OCR đọc bổ sung";
              figure.append(preview); region.append(figure);
            }
            const text = document.createElement("ol");
            for (const row of read.text_lines || []) {
              const item = document.createElement("li"); item.textContent = row.text; text.append(item);
            }
            if (!text.children.length) {
              const empty = document.createElement("p"); empty.textContent = "Chưa đọc được dòng chữ nào trong vùng này."; region.append(empty);
            }
            region.append(text); section.append(region);
          }
          if (report.ocr[side].error) {
            const message = document.createElement("p");
            message.textContent = report.ocr[side].error;
            section.append(message);
          }
          columns.append(section);
        }
        details.append(columns);
      }
      const json = document.createElement("pre");
      json.className = "ocr-lab-json";
      json.textContent = JSON.stringify(report, null, 2);
      details.append(json);
      el("ocr-lab-details").append(details);
    }
    controls();
  }
  function reset() {
    for (const engine of engines) { delete reports[engine.id]; delete elapsed[engine.id]; }
    el("ocr-lab-status").textContent = files.front && files.back ? "Hai ảnh đã sẵn sàng." : "Chọn đủ mặt trước và mặt sau để bắt đầu.";
    render();
  }
  function choose(side, file) {
    if (running || !file) return;
    el(`ocr-lab-${side}-input`).value = "";
    if (!file.size) return error("Ảnh đang trống. Hãy chọn lại ảnh.");
    if (file.size > 10 * 1024 * 1024) return error("Mỗi ảnh tối đa 10 MB.");
    if (!["image/jpeg", "image/png"].includes(file.type)) return error("Chỉ hỗ trợ ảnh JPG hoặc PNG.");
    if (urls[side]) URL.revokeObjectURL(urls[side]);
    files[side] = file;
    urls[side] = URL.createObjectURL(file);
    el(`ocr-lab-${side}-image`).src = urls[side];
    el(`ocr-lab-${side}-filename`).textContent = file.name;
    el(`ocr-lab-${side}-preview`).hidden = false;
    el(`ocr-lab-${side}-dropzone`).hidden = true;
    error("");
    reset();
  }
  for (const side of ["front", "back"]) {
    const card = document.createElement("div");
    card.className = "card-upload";
    card.innerHTML = `<h3 class="upload-side-title">${sideNames[side]}<span>Bắt buộc</span></h3>
      <input id="ocr-lab-${side}-input" class="visually-hidden" type="file" accept="image/jpeg,image/png" aria-label="Chọn ảnh ${sideNames[side].toLowerCase()} thử nghiệm OCR">
      <div id="ocr-lab-${side}-dropzone" class="dropzone"><svg class="icon upload-side-icon" aria-hidden="true"><use href="#i-upload"/></svg><p>Kéo thả ảnh ${sideNames[side].toLowerCase()}</p><button id="ocr-lab-${side}-choose" type="button" class="button button-outline">Chọn ảnh</button><span class="file-limit">JPG, PNG · Tối đa 10 MB</span></div>
      <div id="ocr-lab-${side}-preview" class="preview" hidden><div class="preview-image-wrap"><img id="ocr-lab-${side}-image" alt="Ảnh ${sideNames[side].toLowerCase()} thử nghiệm"></div><div class="file-details"><span class="file-description"><strong id="ocr-lab-${side}-filename"></strong></span><button id="ocr-lab-${side}-remove" type="button" class="icon-button" aria-label="Bỏ ảnh ${sideNames[side].toLowerCase()}"><svg class="icon"><use href="#i-x"/></svg></button></div></div>`;
    el("ocr-lab-uploads").append(card);
    el(`ocr-lab-${side}-choose`).addEventListener("click", () => el(`ocr-lab-${side}-input`).click());
    el(`ocr-lab-${side}-input`).addEventListener("change", event => choose(side, event.target.files[0]));
    el(`ocr-lab-${side}-remove`).addEventListener("click", () => {
      if (running) return;
      files[side] = null;
      if (urls[side]) URL.revokeObjectURL(urls[side]);
      urls[side] = null;
      el(`ocr-lab-${side}-image`).removeAttribute("src");
      el(`ocr-lab-${side}-preview`).hidden = true;
      el(`ocr-lab-${side}-dropzone`).hidden = false;
      error(""); reset();
    });
    const zone = el(`ocr-lab-${side}-dropzone`);
    for (const name of ["dragenter", "dragover", "dragleave", "drop"]) zone.addEventListener(name, event => {
      event.preventDefault();
      zone.classList.toggle("dragover", !running && ["dragenter", "dragover"].includes(name));
      if (name === "drop") choose(side, event.dataTransfer.files[0]);
    });
  }
  for (const engine of engines) {
    const label = document.createElement("label");
    label.innerHTML = `<input type="radio" name="ocr-lab-engine" value="${engine.id}"${engine.id === "namlhn_gpu" ? " checked" : ""}><span><strong>${engine.label}</strong><small id="ocr-lab-health-${engine.id}">${engine.model} · Đang kiểm tra…</small></span>`;
    el("ocr-lab-engines").append(label);
    const summary = document.createElement("section");
    summary.className = "panel ocr-lab-summary";
    summary.innerHTML = `<h3>${engine.label}</h3><span id="ocr-lab-summary-status-${engine.id}" class="qr-method-status">Chưa chạy</span><p id="ocr-lab-summary-info-${engine.id}"></p><p id="ocr-lab-summary-warnings-${engine.id}" class="ocr-lab-summary-warnings"></p>`;
    el("ocr-lab-summaries").append(summary);
  }

  async function run(selected) {
    if (running || !files.front || !files.back) return;
    running = true;
    error("");
    controls();
    for (const engine of selected) {
      active = engine.id;
      delete reports[engine.id]; delete elapsed[engine.id];
      el("ocr-lab-status").textContent = `Đang chạy ${engine.label}…`;
      render();
      const started = performance.now();
      try {
        const form = new FormData();
        for (const side of ["front", "back"]) form.append(`${side}_image`, files[side]);
        form.append("engine", engine.id);
        form.append("use_qr", String(el("ocr-lab-use-qr").checked));
        form.append("include_ocr_text", "true");
        const response = await fetch("/api/cccd/experiment/extract", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok || !data.success) {
          reports[engine.id] = { success: false, error: typeof data.error === "string" ? data.error : "Chưa xử lý được ảnh.", request_id: data.request_id };
        } else reports[engine.id] = data;
      } catch (err) {
        reports[engine.id] = { success: false, error: err.message || "Không kết nối được dịch vụ." };
      }
      elapsed[engine.id] = performance.now() - started;
      active = null;
      render();
    }
    running = false;
    el("ocr-lab-status").textContent = "Đã hoàn tất. Có thể chạy lại từng bộ xử lý trên cùng hai ảnh.";
    render();
  }
  el("ocr-lab-run").addEventListener("click", () => run(engines.filter(engine => engine.id === document.querySelector('input[name="ocr-lab-engine"]:checked').value)));
  el("ocr-lab-run-all").addEventListener("click", () => run(engines));
  el("ocr-lab-run-docu").addEventListener("click", () => run(engines.filter(engine => engine.id.startsWith("docu_gpu"))));
  el("ocr-lab-use-qr").addEventListener("change", reset);
  const exported = () => JSON.stringify({ experiment: "full_ocr_comparison", results: reports }, null, 2);
  el("ocr-lab-copy").addEventListener("click", () => copyText(exported()));
  el("ocr-lab-download").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([exported()], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "ocr-comparison.json"; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  fetch("/api/cccd/experiment/health").then(response => response.json()).then(data => {
    for (const engine of engines) {
      const info = data.engines[engine.id];
      const ready = info.available ?? info.source_available;
      el(`ocr-lab-health-${engine.id}`).textContent = `${info.recognition_model || info.model || engine.model}${info.strategy === "namlhn_docu_v1" ? " · bộ đọc DocU" : info.strategy === "docu_regions_v2" ? " · v2" : info.strategy === "docu_expanded_v3" ? " · v3" : ""}${info.decoder === "cuda_batched_v1" ? " · batch GPU" : ""} · ${ready ? info.model_loaded ? "Đã nạp model" : "Sẵn sàng, chưa nạp model" : "Chưa sẵn sàng"}`;
    }
  }).catch(() => {
    for (const engine of engines) el(`ocr-lab-health-${engine.id}`).textContent = `${engine.model} · Chưa kiểm tra được`;
  });
  reset();
})();
