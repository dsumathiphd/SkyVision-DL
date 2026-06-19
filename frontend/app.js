const form = document.querySelector("#inspection-form");
const statusBox = document.querySelector("#status");
const results = document.querySelector("#results");
const summaryGrid = document.querySelector("#summary-grid");
const framesContainer = document.querySelector("#frames");
const submitButton = document.querySelector("#submit-button");
const jsonReport = document.querySelector("#json-report");
const htmlReport = document.querySelector("#html-report");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Running inspection...", false);
  setBusy(true);
  results.hidden = true;

  const formData = new FormData(form);
  if (!formData.get("model_path")) {
    formData.delete("model_path");
  }

  try {
    const response = await fetch("/api/inspections", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Inspection failed.");
    }

    renderResults(payload);
    setStatus("Inspection complete.", false);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
});

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "Inspecting..." : "Run Inspection";
}

function setStatus(message, isError) {
  statusBox.hidden = false;
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function renderResults(payload) {
  const summary = payload.summary;
  const classCounts = summary.detections_by_class || {};
  const topClasses = Object.entries(classCounts)
    .map(([name, count]) => `${name}: ${count}`)
    .join(", ") || "None";

  summaryGrid.innerHTML = "";
  addMetric("Frames", summary.total_frames_sampled);
  addMetric("Detections", summary.total_detections);
  addMetric("FPS", Number(summary.video_metadata.fps).toFixed(2));
  addMetric("Classes", topClasses);

  jsonReport.href = payload.report.json_url;
  htmlReport.href = payload.report.html_url;

  framesContainer.innerHTML = "";
  payload.frames.forEach((frame, index) => {
    framesContainer.appendChild(renderFrame(frame, index));
  });

  results.hidden = false;
}

function addMetric(label, value) {
  const metric = document.createElement("div");
  metric.className = "metric";
  metric.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong>`;
  summaryGrid.appendChild(metric);
}

function renderFrame(frame, index) {
  const card = document.createElement("article");
  card.className = "frame-card";

  const detections = frame.detections.length
    ? frame.detections
      .map((item) => `<li>${escapeHtml(item.class_name)} ${Number(item.confidence).toFixed(2)}</li>`)
      .join("")
    : "<li>No defects</li>";

  card.innerHTML = `
    <img src="${frame.annotated_image_url}" alt="Annotated frame ${index + 1}">
    <div class="frame-body">
      <h3>Frame ${index + 1}</h3>
      <ul class="detections">${detections}</ul>
    </div>
  `;

  return card;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

