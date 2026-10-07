/* View — hanya berbicara ke API (arsitektur.md §3.1).
   Tidak ada preprocessing/prediksi di sisi klien. */

"use strict";

const $ = (id) => document.getElementById(id);

const state = { file: null };

const MAX_MB = 10;
const OK_TYPES = ["image/jpeg", "image/png", "image/webp"];
const OK_EXT = [".jpg", ".jpeg", ".png", ".webp"];

/* Nama tampilan untuk label kelas.
   API tetap membalas nama kelas apa adanya (== kunci `class_indices.json`),
   pemetaan ini murni kosmetik di View agar sama dengan teks §2.2. */
const LABEL_DISPLAY = {
  DALLE: "DALL·E",
  "DALL·E": "DALL·E",
  "DALL-E": "DALL·E",
  GEMINI: "Gemini",
  MIDJOURNEY: "Midjourney",
  STABLE_D: "Stable Diffusion",
  STABLE_DIFFUSION: "Stable Diffusion",
};

function prettyLabel(label) {
  if (!label) return label;
  const hit = LABEL_DISPLAY[String(label).toUpperCase()];
  return hit || label;
}

/* ------------------------- ketentuan (§2.2) --------------------------- */

async function loadRequirements() {
  try {
    const res = await fetch("/api/v1/requirements");
    if (!res.ok) return;
    const data = await res.json();

    const setList = (id, items) => {
      const el = $(id);
      if (!items || !items.length) return;
      el.innerHTML = "";
      items.forEach((text) => {
        const li = document.createElement("li");
        li.textContent = text;
        el.appendChild(li);
      });
    };

    if (data.intro) $("terms-intro").textContent = data.intro;
    setList("terms-allowed", data.allowed);
    setList("terms-prohibited", data.prohibited);
    if (data.warning) $("terms-warning").textContent = "⚠️ " + data.warning;
    if (data.agree_text) $("agree-text").textContent = data.agree_text;
    if (data.technical) {
      const t = data.technical;
      $("terms-technical").textContent =
        `Format: ${t.formats.join(" / ")} · maks ${t.max_size_mb} MB · ` +
        `sisi terpendek ≥ ${t.min_side_px} px`;
    }
  } catch (err) {
    console.warn("Ketentuan tidak bisa dimuat, memakai teks bawaan.", err);
  }
}

/* ----------------------------- kesehatan ------------------------------ */

async function checkHealth() {
  const banner = $("health-banner");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.model_loaded && data.face_detector_loaded && data.labels_loaded) {
      banner.hidden = true;
      return;
    }
    banner.hidden = false;
    banner.textContent =
      "⚠️ Komponen belum siap" +
      (data.startup_error ? ` — ${data.startup_error}` : "") +
      " (lihat Fase 0 pada README).";
  } catch (err) {
    banner.hidden = false;
    banner.textContent = "⚠️ Tidak bisa menghubungi server.";
  }
}

/* ------------------------------- upload ------------------------------- */

function acceptFile(file) {
  const status = $("upload-status");
  status.classList.remove("error-text");

  const ext = (file.name || "").toLowerCase().slice(file.name.lastIndexOf("."));
  const typeOk = OK_TYPES.includes(file.type) || OK_EXT.includes(ext);
  if (!typeOk) {
    setStatus("Format tidak didukung. Gunakan JPG, PNG, atau WEBP.", true);
    return false;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    setStatus(`Ukuran file melebihi ${MAX_MB} MB.`, true);
    return false;
  }

  state.file = file;
  $("preview").src = URL.createObjectURL(file);
  $("preview-wrap").hidden = false;
  $("dropzone").hidden = true;
  setStatus(`Terpilih: ${file.name}`);
  refreshSubmit();
  return true;
}

function clearFile() {
  if (state.file) URL.revokeObjectURL($("preview").src);
  state.file = null;
  $("file-input").value = "";
  $("preview-wrap").hidden = true;
  $("dropzone").hidden = false;
  setStatus("");
  refreshSubmit();
}

function setStatus(text, isError = false) {
  const el = $("upload-status");
  el.textContent = text;
  el.classList.toggle("error-text", Boolean(isError));
}

function refreshSubmit() {
  $("submit-btn").disabled = !(state.file && $("agree").checked);
}

/* ------------------------------ prediksi ------------------------------ */

function showResult(payload) {
  $("result-empty").hidden = true;
  $("result-error").hidden = true;
  $("result-body").hidden = false;

  const p = payload.prediction;
  $("result-label").textContent = prettyLabel(p.label);
  $("result-confidence").textContent = (p.confidence * 100).toFixed(1) + "%";

  const notes = {
    high: "Keyakinan model tinggi.",
    medium: "⚠️ Keyakinan sedang — hasil sebaiknya diverifikasi ulang.",
    uncertain:
      "⚠️ Keyakinan rendah: gambar tidak cukup meyakinkan, label ditahan (unknown).",
  };
  $("result-note").textContent = notes[p.status] || "";

  const bars = $("bars");
  bars.innerHTML = "";
  p.top_k.forEach((item) => {
    const pct = item.confidence * 100;
    const row = document.createElement("div");
    row.className = "bar-row" + (item.label === p.label ? " top" : "");
    row.innerHTML =
      `<span class="name"></span>` +
      `<span class="bar-track"><span class="bar-fill" style="width:0%"></span></span>` +
      `<span class="pct"></span>`;
    row.querySelector(".name").textContent = prettyLabel(item.label);
    row.querySelector(".pct").textContent = pct.toFixed(1) + "%";
    bars.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = pct.toFixed(1) + "%";
    });
  });

  const fc = payload.face_check;
  const coverage =
    fc.face_coverage != null ? ` · cakupan wajah ${(fc.face_coverage * 100).toFixed(0)}%` : "";
  $("face-check").textContent =
    `✅ ${fc.faces_detected} wajah terdeteksi${coverage} — sesuai ketentuan (${fc.detector})`;

  const m = payload.model;
  $("result-meta").textContent =
    `model ${m.name} v${m.version}` +
    (m.keras_version ? ` · keras ${m.keras_version}` : "") +
    ` · ${payload.processing_time_ms} ms · request ${payload.request_id.slice(0, 8)}`;
}

function showError(error) {
  $("result-empty").hidden = true;
  $("result-body").hidden = true;
  $("result-error").hidden = false;

  $("error-code").textContent = `DITOLAK — ${error.code || "ERROR"}`;
  $("error-message").textContent = error.message || "Terjadi kesalahan.";
  $("error-ref").textContent = error.requirement_ref
    ? `Lihat ketentuan: ${error.requirement_ref}`
    : "";
}

function resetResult() {
  $("result-body").hidden = true;
  $("result-error").hidden = true;
  $("result-empty").hidden = false;
}

async function submit() {
  if (!state.file) return;

  const btn = $("submit-btn");
  btn.disabled = true;
  btn.textContent = "Memproses…";
  resetResult();
  setStatus("Mengunggah & memeriksa ketentuan…");

  const form = new FormData();
  form.append("file", state.file, state.file.name);
  form.append("top_k", "4");
  form.append("agree_terms", String($("agree").checked));

  try {
    const res = await fetch("/api/v1/predict", { method: "POST", body: form });
    const payload = await res.json().catch(() => ({}));

    if (res.ok && payload.success) {
      showResult(payload);
      setStatus("Selesai.");
    } else {
      showError(payload.error || { code: `HTTP_${res.status}`, message: "Permintaan gagal." });
      setStatus("");
    }
  } catch (err) {
    showError({ code: "NETWORK_ERROR", message: "Tidak bisa terhubung ke server." });
    setStatus("");
  } finally {
    btn.textContent = "Klasifikasi";
    refreshSubmit();
  }
}

/* ------------------------------- wiring ------------------------------- */

function init() {
  const dz = $("dropzone");
  const input = $("file-input");

  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => input.files[0] && acceptFile(input.files[0]));

  ["dragenter", "dragover"].forEach((evt) =>
    dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.remove("dragover"); })
  );
  dz.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (file) acceptFile(file);
  });

  $("agree").addEventListener("change", refreshSubmit);
  $("submit-btn").addEventListener("click", submit);
  $("clear-btn").addEventListener("click", () => { clearFile(); resetResult(); });
  $("retry-btn").addEventListener("click", () => { clearFile(); resetResult(); });

  loadRequirements();
  checkHealth();
}

document.addEventListener("DOMContentLoaded", init);
