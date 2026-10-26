"use strict";

(() => {
  const el = id => document.getElementById(id);
  const methods = [
    { id: "zxing", label: "ZXing", description: "Tìm và giải mã trực tiếp trên ảnh gốc." },
    { id: "wechat", label: "WeChatQRCode", description: "Phát hiện QR, tăng độ phân giải và giải mã bằng WeChat." },
    { id: "qrdet", label: "QRDet + ZXing", description: "QRDet small tìm vùng QR; nắn phối cảnh, thêm viền trắng rồi dùng ZXing giải mã." },
    { id: "wechat_restormer", label: "WeChat + Restormer", description: "Tìm vùng QR, khử nhòe bằng Restormer trên GPU rồi đọc bằng WeChat." },
    { id: "wechat_swinir", label: "WeChat + SwinIR", description: "Tìm vùng QR, tăng độ phân giải x4 bằng SwinIR-S trên GPU rồi đọc bằng WeChat." },
    { id: "wechat_both", label: "WeChat + Restormer + SwinIR", description: "Khử nhòe bằng Restormer, tăng độ phân giải x4 bằng SwinIR-S, sau đó WeChat giải mã." },
  ];
  const qrFields = [
    ["id_number", "Số CCCD"], ["old_id_number", "Số CMND cũ"], ["name", "Họ và tên"],
    ["date_of_birth", "Ngày sinh"], ["gender", "Giới tính"],
    ["place_of_residence", "Nơi cư trú"], ["issue_date", "Ngày cấp"],
  ];
  const statuses = { complete: "Đọc đủ thông tin", partial: "Đọc được một phần", no_qr: "Chưa đọc được QR",
    detected_not_decoded: "Có vùng QR, chưa giải mã", multiple_qr: "Nhiều nội dung QR", unsupported_qr: "QR không có cấu trúc CCCD",
    partial_error: "Một mặt xử lý lỗi" };
  const reports = {};
  const files = { front: null, back: null };
  const previewURLs = { front: null, back: null };
  const sideNames = { front: "Mặt trước", back: "Mặt sau" };
  const uploadID = (side, suffix) => `qr-${side === "back" ? "back-" : ""}${suffix}`;
  const pairMode = () => el("qr-scope").querySelector('input[value="pair"]').checked;
  const executionMode = () => el("qr-execution").querySelector("input:checked").value;
  let running = false;
  const ms = value => `${Number(value).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} ms`;

  const workspaces = ["extract", "qr", "experiment"];
  function setWorkspace(name, focus = false) {
    for (const id of workspaces) {
      const selected = id === name;
      el(`workspace-tab-${id}`).setAttribute("aria-selected", String(selected));
      el(`workspace-tab-${id}`).tabIndex = selected ? 0 : -1;
      el(`workspace-${id}`).hidden = !selected;
    }
    if (focus) el(`workspace-tab-${name}`).focus();
    history.replaceState(null, "", location.pathname + location.search + (name === "extract" ? "" : `#${name}`));
  }
  for (const name of workspaces) {
    el(`workspace-tab-${name}`).addEventListener("click", () => setWorkspace(name));
    el(`workspace-tab-${name}`).addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const index = workspaces.indexOf(name);
      const next = event.key === "Home" ? 0 : event.key === "End" ? workspaces.length - 1 :
        (index + (event.key === "ArrowRight" ? 1 : -1) + workspaces.length) % workspaces.length;
      setWorkspace(workspaces[next], true);
    });
  }
  if (workspaces.includes(location.hash.slice(1))) setWorkspace(location.hash.slice(1));
  if (location.hash === "#expiry") setWorkspace("experiment");

  for (const method of methods) {
    const card = document.createElement("section");
    card.className = "panel qr-method";
    card.id = `qr-method-${method.id}`;
    card.setAttribute("aria-labelledby", `qr-title-${method.id}`);
    card.innerHTML = `<div class="qr-method-header"><h3 id="qr-title-${method.id}">${method.label}</h3><span class="qr-method-status" id="qr-status-${method.id}">Chưa chạy</span></div>
      <p class="qr-method-description">${method.description}</p>
      <p class="qr-availability" id="qr-availability-${method.id}">Đang kiểm tra bộ đọc…</p>
      <button type="button" id="qr-run-${method.id}" class="button button-outline" disabled>Chạy ${method.label}</button>
      <div id="qr-output-${method.id}" class="qr-method-output" aria-live="polite"><p>Chưa có kết quả trên ảnh này.</p></div>
      <canvas id="qr-canvas-${method.id}" class="qr-canvas" role="img" aria-label="Vùng QR do ${method.label} trả về" hidden></canvas>`;
    el("qr-methods").append(card);
    el(`qr-run-${method.id}`).addEventListener("click", () => run([method]));
  }

  function error(message) {
    el("qr-upload-error").textContent = message;
    el("qr-upload-error").hidden = !message;
  }
  function updateControls() {
    const ready = files.front && (!pairMode() || files.back);
    el("qr-run-all").disabled = running || !ready;
    el("qr-run-all").classList.toggle("processing", running);
    el("qr-run-all").querySelector(".spinner").hidden = !running;
    el("qr-run-all").querySelector(".icon").hidden = running;
    for (const side of ["front", "back"]) {
      for (const suffix of ["file-input", "choose-file", "remove-file"]) el(uploadID(side, suffix)).disabled = running;
    }
    el("qr-scope").disabled = running;
    el("qr-execution").disabled = running;
    for (const method of methods) el(`qr-run-${method.id}`).disabled = running || !ready;
    el("qr-methods").setAttribute("aria-busy", String(running));
  }
  function clearReports() {
    for (const method of methods) {
      delete reports[method.id];
      el(`qr-status-${method.id}`).textContent = "Chưa chạy";
      el(`qr-status-${method.id}`).className = "qr-method-status";
      el(`qr-output-${method.id}`).replaceChildren();
      const note = document.createElement("p");
      note.textContent = "Chưa có kết quả trên ảnh này.";
      el(`qr-output-${method.id}`).append(note);
      el(`qr-canvas-${method.id}`).hidden = true;
    }
    renderComparison();
  }
  function readyMessage() {
    const ready = files.front && (!pairMode() || files.back);
    el("qr-run-status").textContent = ready ? "Ảnh đã sẵn sàng. Chạy từng phương án hoặc cả sáu." :
      pairMode() ? "Chọn đủ mặt trước và mặt sau để bắt đầu." : "Chọn một ảnh để bắt đầu.";
  }
  function chooseFile(selected, side = "front") {
    if (running || !selected) return;
    el(uploadID(side, "file-input")).value = "";
    if (!selected.size) return error("Ảnh đang trống. Hãy chọn lại ảnh.");
    if (selected.size > 10 * 1024 * 1024) return error("Mỗi ảnh tối đa 10 MB.");
    if (!["image/jpeg", "image/png"].includes(selected.type)) return error("Chỉ hỗ trợ ảnh JPG hoặc PNG.");
    if (previewURLs[side]) URL.revokeObjectURL(previewURLs[side]);
    files[side] = selected;
    previewURLs[side] = URL.createObjectURL(selected);
    el(uploadID(side, "preview-image")).src = previewURLs[side];
    el(uploadID(side, "file-name")).textContent = selected.name;
    el(uploadID(side, "file-size")).textContent = `${(selected.size / 1024).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} KB`;
    el(uploadID(side, "dropzone")).hidden = true;
    el(uploadID(side, "preview")).hidden = false;
    readyMessage();
    error("");
    clearReports();
    updateControls();
  }
  function clearFile(side = "front") {
    if (running) return;
    files[side] = null;
    if (previewURLs[side]) URL.revokeObjectURL(previewURLs[side]);
    previewURLs[side] = null;
    el(uploadID(side, "preview-image")).removeAttribute("src");
    el(uploadID(side, "file-input")).value = "";
    el(uploadID(side, "dropzone")).hidden = false;
    el(uploadID(side, "preview")).hidden = true;
    readyMessage();
    error("");
    clearReports();
    updateControls();
  }

  function drawRegions(method, side = null) {
    const report = side ? reports[method.id]?.sides?.[side] : reports[method.id];
    const canvas = el(`qr-canvas-${method.id}${side ? `-${side}` : ""}`);
    if (!canvas) return;
    const image = el(uploadID(side || "front", "preview-image"));
    if (report?.sides) { canvas.hidden = true; return; }
    if (!report?.success || !image.complete || !image.naturalWidth) { canvas.hidden = true; return; }
    const width = report.image.width, height = report.image.height;
    const scale = Math.min(1, 900 / Math.max(width, height));
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    for (const region of report.regions) {
      if (!region.points?.length) continue;
      ctx.strokeStyle = region.decoded ? "#159669" : "#d98c16";
      ctx.lineWidth = 3;
      ctx.beginPath();
      region.points.forEach(([x, y], index) => index ? ctx.lineTo(x * scale, y * scale) : ctx.moveTo(x * scale, y * scale));
      ctx.closePath();
      ctx.stroke();
    }
    canvas.hidden = false;
  }

  function renderReport(method, expandSides = true) {
    const report = reports[method.id];
    const status = el(`qr-status-${method.id}`);
    status.textContent = report.success ? statuses[report.status] || report.status : "Lỗi";
    status.className = `qr-method-status ${report.success && report.decoded_count && !["multiple_qr", "partial_error"].includes(report.status) ? "success" : "attention"}`;
    const output = el(`qr-output-${method.id}`);
    output.replaceChildren();
    if (report.sides) {
      const summary = document.createElement("p");
      summary.className = "qr-pair-summary";
      summary.textContent = `${report.execution === "parallel" ? "Song song" : "Tuần tự"} · Tổng server: ${ms(report.processing_time_ms)} · Tổng chờ: ${ms(report.client_elapsed_ms)}. Thực thi hai mặt: ${ms(report.timings_ms.execution)}; chờ lượt: ${ms(report.timings_ms.queue_wait)}. Tổng thời gian từng mặt: ${ms(report.timings_ms.sum_side_processing)} (có thể chồng nhau).`;
      output.append(summary);
      const message = document.createElement("p");
      message.textContent = report.message;
      output.append(message);
      for (const side of ["front", "back"]) {
        const section = document.createElement("details");
        section.className = "qr-side-result";
        section.open = expandSides;
        const heading = document.createElement("summary");
        const sideReport = report.sides[side];
        heading.textContent = `${sideNames[side]} · ${sideReport.success ? statuses[sideReport.status] || sideReport.status : "Lỗi"} · ${ms(sideReport.processing_time_ms)}`;
        section.append(heading);
        const schedule = document.createElement("p");
        schedule.className = "qr-metric-note";
        schedule.textContent = `Bắt đầu: ${ms(sideReport.schedule_ms.started_at_ms)} · Kết thúc: ${ms(sideReport.schedule_ms.finished_at_ms)} từ lúc chạy hai mặt.`;
        section.append(schedule);
        renderSingleReport(sideReport, section);
        if (sideReport.success) {
          const canvas = document.createElement("canvas");
          canvas.id = `qr-canvas-${method.id}-${side}`;
          canvas.className = "qr-canvas";
          canvas.setAttribute("role", "img");
          canvas.setAttribute("aria-label", `Vùng QR ${sideNames[side].toLowerCase()} do ${method.label} trả về`);
          section.append(canvas);
        }
        output.append(section);
        drawRegions(method, side);
      }
      return;
    }
    renderSingleReport(report, output);
    drawRegions(method);
  }

  function renderSingleReport(report, output) {
    const message = document.createElement("p");
    message.textContent = report.success ? report.message : report.error;
    output.append(message);
    if (report.success) {
      const metrics = document.createElement("dl");
      const add = (name, value) => {
        const row = document.createElement("div");
        const term = document.createElement("dt"), detail = document.createElement("dd");
        term.textContent = name;
        detail.textContent = value;
        row.append(term, detail);
        metrics.append(row);
      };
      add("Nội dung QR đã giải mã", report.decoded_count);
      add("Trường thông tin", `${report.recognized_fields} / ${report.total_fields}`);
      add("Đọc QR", ms(report.timings_ms.detect_decode));
      add("Nạp bộ đọc / model", ms(report.timings_ms.initialize));
      add("Xử lý ảnh trên server", ms(report.processing_time_ms));
      if (report.client_elapsed_ms !== undefined) add("Tổng chờ", ms(report.client_elapsed_ms));
      if (report.restoration) {
        const data = report.restoration;
        add("Vùng QR tìm được / đã xử lý", `${data.detected_regions} / ${data.processed_regions}`);
        add("Thiết bị phục hồi", "GPU");
        add("Tìm vùng QR", ms(data.timings_ms.detect_regions));
        add("Phục hồi và nhận ảnh từ GPU", ms(data.timings_ms.restore));
        add("WeChat giải mã vùng phục hồi", ms(data.timings_ms.decode));
        const crops = Array.isArray(data.crops) ? data.crops : [];
        const initializeMs = crops.reduce((sum, crop) => sum + (crop.initialize_ms || 0), 0);
        add("Nạp model phục hồi", ms(initializeMs));
        for (const name of ["restormer", "swinir"]) {
          const stages = crops.flatMap(crop => crop.steps || []).filter(step => step.model === name);
          if (stages.length) add(name === "restormer" ? "Restormer khử nhòe" : "SwinIR tăng độ phân giải", ms(stages.reduce((sum, step) => sum + step.processing_time_ms, 0)));
        }
        const peak = Math.max(0, ...crops.map(crop => crop.gpu_peak_allocated_mb || 0));
        if (peak) add("VRAM PyTorch tối đa (worker)", `${peak.toLocaleString("vi-VN")} MB`);
        if (crops.length) add("Kích thước vùng trước → sau", `${crops[0].input_size.join(" × ")} → ${crops[0].output_size.join(" × ")} px`);
      }
      output.append(metrics);
      const note = document.createElement("p");
      note.className = "qr-metric-note";
      note.textContent = `${report.cold_start ? "Lần nạp bộ đọc đầu tiên. " : "Bộ đọc đã được nạp. "}Khung xanh: đã giải mã; vàng: chưa giải mã. Nạp model và đọc QR nằm trong tổng xử lý server.`;
      output.append(note);
      if (report.restoration) {
        const branchNote = document.createElement("p");
        branchNote.className = "qr-metric-note";
        branchNote.textContent = "Ba phép thử phục hồi dùng cùng cách tìm vùng QR. Chỉ giải mã ảnh đã phục hồi, không lấy kết quả ảnh gốc thay thế. Thời gian model nằm trong tổng phục hồi; VRAM gồm các model đang nạp trong worker.";
        output.append(branchNote);
        if (report.restoration.truncated) {
          const limit = document.createElement("p");
          limit.textContent = `Ảnh có nhiều vùng QR; chỉ xử lý ${report.restoration.region_limit} vùng có độ tin cậy cao nhất.`;
          output.append(limit);
        }
        const previews = report.restoration.previews;
        if (previews) {
          const container = document.createElement("div");
          container.className = "qr-restoration-previews";
          for (const [key, label] of [["before", "Vùng QR trước"], ["after", "Sau phục hồi"]]) {
            if (typeof previews[key] !== "string" || !previews[key].startsWith("data:image/jpeg;base64,")) continue;
            const figure = document.createElement("figure");
            const image = document.createElement("img");
            image.src = previews[key];
            image.alt = label;
            const caption = document.createElement("figcaption");
            caption.textContent = label;
            figure.append(image, caption);
            container.append(figure);
          }
          output.append(container);
        }
      }
    }
  }

  function renderComparison() {
    const body = el("qr-comparison-rows");
    body.replaceChildren();
    const comparisonSides = pairMode() ? ["front", "back"] : [null];
    for (const side of comparisonSides) for (const [key, label] of qrFields) {
      const row = document.createElement("tr");
      const heading = document.createElement("th");
      heading.scope = "row";
      heading.textContent = side ? `${sideNames[side]} · ${label}` : label;
      row.append(heading);
      const values = methods.map(method => (side ? reports[method.id]?.sides?.[side] : reports[method.id])?.fields?.[key] || "");
      const differs = new Set(values.filter(Boolean)).size > 1;
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value || "—";
        if (value && differs) cell.className = "qr-value-differs";
        row.append(cell);
      }
      body.append(row);
    }
    const empty = !Object.keys(reports).length;
    el("qr-copy-json").disabled = empty;
    el("qr-download-json").disabled = empty;
  }

  async function run(selectedMethods) {
    if (running || !files.front || (pairMode() && !files.back)) return;
    running = true;
    updateControls();
    error("");
    try {
      for (const method of selectedMethods) {
        delete reports[method.id];
        renderComparison();
        el(`qr-status-${method.id}`).textContent = "Đang chạy…";
        el(`qr-status-${method.id}`).className = "qr-method-status";
        el(`qr-output-${method.id}`).textContent = "Đang xử lý ảnh. Lần đầu có thể cần thêm thời gian nạp model.";
        el(`qr-canvas-${method.id}`).hidden = true;
        el("qr-run-status").textContent = `Đang chạy ${method.label} ${pairMode() ? "trên hai mặt" : "trên ảnh đã chọn"}…`;
        const started = performance.now();
        const form = new FormData();
        if (pairMode()) {
          form.append("front_image", files.front);
          form.append("back_image", files.back);
          form.append("execution", executionMode());
        } else form.append("file", files.front);
        form.append("method", method.id);
        try {
          const response = await fetch(pairMode() ? "/api/qr/experiment/pair" : "/api/qr/experiment", { method: "POST", body: form });
          let payload;
          try { payload = await response.json(); }
          catch { throw new Error("Dịch vụ trả về kết quả không hợp lệ. Hãy thử lại."); }
          if (!response.ok || !payload.success) throw new Error(payload.error || "Không chạy được phương án QR này.");
          if (payload.method !== method.id || !payload.fields || !payload.timings_ms ||
              (pairMode() ? !payload.sides?.front || !payload.sides?.back : !Array.isArray(payload.regions) || !payload.image)) throw new Error("Kết quả thử nghiệm thiếu thông tin.");
          reports[method.id] = { ...payload, client_elapsed_ms: Math.round((performance.now() - started) * 100) / 100 };
        } catch (problem) {
          reports[method.id] = { success: false, method: method.id,
            error: problem instanceof TypeError ? "Mất kết nối với dịch vụ. Hãy thử lại." : problem.message };
        }
        renderReport(method, selectedMethods.length === 1);
        renderComparison();
      }
      el("qr-run-status").textContent = "Đã chạy xong. Bạn có thể thử lại từng phương án trên cùng ảnh.";
    } finally {
      running = false;
      updateControls();
    }
  }

  for (const side of ["front", "back"]) {
    el(uploadID(side, "choose-file")).addEventListener("click", () => el(uploadID(side, "file-input")).click());
    el(uploadID(side, "file-input")).addEventListener("change", event => chooseFile(event.target.files[0], side));
    el(uploadID(side, "remove-file")).addEventListener("click", () => clearFile(side));
    el(uploadID(side, "preview-image")).addEventListener("load", () => methods.forEach(method => {
      drawRegions(method);
      drawRegions(method, side);
    }));
    el(uploadID(side, "preview-image")).addEventListener("error", () => {
      clearFile(side);
      error("Không xem trước được ảnh. Hãy chọn lại ảnh JPG hoặc PNG hợp lệ.");
    });
    for (const name of ["dragenter", "dragover", "dragleave", "drop"]) {
      el(uploadID(side, "dropzone")).addEventListener(name, event => {
        event.preventDefault();
        if (running) return;
        el(uploadID(side, "dropzone")).classList.toggle("dragover", ["dragenter", "dragover"].includes(name));
        if (name === "drop") chooseFile(event.dataTransfer.files[0], side);
      });
    }
  }
  el("qr-scope").addEventListener("change", () => {
    el("qr-back-upload").hidden = !pairMode();
    el("qr-execution").hidden = !pairMode();
    el("qr-front-title").textContent = pairMode() ? "Mặt trước" : "Ảnh QR";
    el("qr-preview-image").alt = pairMode() ? "Ảnh mặt trước dùng thử nghiệm QR" : "Ảnh dùng thử nghiệm QR";
    error("");
    clearReports();
    readyMessage();
    updateControls();
  });
  el("qr-run-all").addEventListener("click", () => run(methods));
  const exportText = () => JSON.stringify({
    ...(!pairMode() ? { image: files.front ? { name: files.front.name, size: files.front.size } : null } : {}),
    images: Object.fromEntries((pairMode() ? ["front", "back"] : ["front"]).map(side =>
      [side, files[side] ? { name: files[side].name, size: files[side].size } : null])), results: reports,
  }, null, 2);
  el("qr-copy-json").addEventListener("click", () => copyText(exportText()));
  el("qr-download-json").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([exportText()], { type: "application/json;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "qr-comparison.json";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  window.addEventListener("beforeunload", () => Object.values(previewURLs).forEach(url => { if (url) URL.revokeObjectURL(url); }));
  fetch("/api/health").then(response => {
    if (!response.ok) throw new Error("Health check failed");
    return response.json();
  }).then(health => {
    for (const method of methods) {
      const info = health.qr_experiments?.[method.id];
      el(`qr-availability-${method.id}`).textContent = info?.message || "Chưa có thông tin bộ đọc.";
      el(`qr-availability-${method.id}`).classList.toggle("unavailable", info?.available === false);
    }
  }).catch(() => {
    for (const method of methods) el(`qr-availability-${method.id}`).textContent = "Chưa kiểm tra được bộ đọc. Bạn vẫn có thể thử chạy.";
  });
  renderComparison();
})();
