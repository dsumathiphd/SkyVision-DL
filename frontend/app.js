const form = document.querySelector("#inspection-form");
const analysisModeInputs = document.querySelectorAll("input[name='analysis_mode']");
const videoInput = document.querySelector("#video");
const imageInput = document.querySelector("#image");
const videoUploadField = document.querySelector("#video-upload-field");
const imageUploadField = document.querySelector("#image-upload-field");
const authPanel = document.querySelector("#auth-panel");
const inspectionPanel = document.querySelector("#inspection-panel");
const loginForm = document.querySelector("#login-form");
const registerForm = document.querySelector("#register-form");
const loginTab = document.querySelector("#login-tab");
const registerTab = document.querySelector("#register-tab");
const loginButton = document.querySelector("#login-button");
const registerButton = document.querySelector("#register-button");
const sessionBox = document.querySelector("#session-box");
const sessionUser = document.querySelector("#session-user");
const logoutButton = document.querySelector("#logout-button");
const statusBox = document.querySelector("#status");
const dashboardPanel = document.querySelector("#dashboard-panel");
const jobPanel = document.querySelector("#job-panel");
const jobIdBox = document.querySelector("#job-id");
const jobStatusBox = document.querySelector("#job-status");
const jobProgressBox = document.querySelector("#job-progress");
const jobProgressBar = document.querySelector("#job-progress-bar");
const jobResultState = document.querySelector("#job-result-state");
const jobStageLabel = document.querySelector("#job-stage-label");
const results = document.querySelector("#results");
const summaryGrid = document.querySelector("#summary-grid");
const priorityList = document.querySelector("#priority-list");
const framesContainer = document.querySelector("#frames");
const submitButton = document.querySelector("#submit-button");
const jsonReport = document.querySelector("#json-report");
const htmlReport = document.querySelector("#html-report");
const pdfReport = document.querySelector("#pdf-report");
const zipReport = document.querySelector("#zip-report");
const reportStatus = document.querySelector("#report-status");
const historyPanel = document.querySelector("#history-panel");
const historyBody = document.querySelector("#history-body");
const historyEmpty = document.querySelector("#history-empty");
const refreshHistoryButton = document.querySelector("#refresh-history");
const exportHistoryCsvButton = document.querySelector("#export-history-csv");
const historyFilterButtons = document.querySelectorAll(".history-filter");
const historySearchInput = document.querySelector("#history-search");
const systemPanel = document.querySelector("#system-panel");
const systemStatusGrid = document.querySelector("#system-status-grid");
const refreshSystemButton = document.querySelector("#refresh-system");
const settingsPanel = document.querySelector("#settings-panel");
const settingsForm = document.querySelector("#settings-form");
const saveSettingsButton = document.querySelector("#save-settings");
const settingsStatus = document.querySelector("#settings-status");
const downloadBackupButton = document.querySelector("#download-backup");
const settingsInputs = {
  sampleFps: document.querySelector("#settings-default-sample-fps"),
  confidence: document.querySelector("#settings-default-confidence"),
  device: document.querySelector("#settings-default-device"),
  modelPath: document.querySelector("#settings-default-model-path"),
  maxVideoUploadMb: document.querySelector("#settings-max-video-upload-mb"),
  maxImageUploadMb: document.querySelector("#settings-max-image-upload-mb"),
};
const adminPanel = document.querySelector("#admin-panel");
const adminUsersBody = document.querySelector("#admin-users-body");
const adminEmpty = document.querySelector("#admin-empty");
const refreshUsersButton = document.querySelector("#refresh-users");
const dashboardMetrics = {
  total: document.querySelector("#metric-total-inspections"),
  completed: document.querySelector("#metric-completed"),
  uniqueDefects: document.querySelector("#metric-unique-defects"),
  reportsReady: document.querySelector("#metric-reports-ready"),
  critical: document.querySelector("#metric-critical-defects"),
  high: document.querySelector("#metric-high-defects"),
  reliability: document.querySelector("#metric-reliability"),
  primaryDefect: document.querySelector("#metric-primary-defect"),
};
const dashboardRiskLabel = document.querySelector("#dashboard-risk-label");
const workflowCurrentState = document.querySelector("#workflow-current-state");
const dashboardSettings = {
  sampleFps: document.querySelector("#setting-sample-fps"),
  confidence: document.querySelector("#setting-confidence"),
  device: document.querySelector("#setting-device"),
  modelPath: document.querySelector("#setting-model-path"),
  modelVersion: document.querySelector("#setting-model-version"),
  productModel: document.querySelector("#product-model-label"),
  productConfidence: document.querySelector("#product-confidence-label"),
  productDevice: document.querySelector("#product-device-label"),
};
const workflowSteps = {
  extract: document.querySelector("#step-extract"),
  yolo: document.querySelector("#step-yolo"),
  results: document.querySelector("#step-results"),
  report: document.querySelector("#step-report"),
};

const TOKEN_KEY = "skyvision_access_token";
const USER_KEY = "skyvision_user";
const POLL_INTERVAL_MS = 1500;
const USER_ROLES = ["admin", "inspector", "maintenance_manager", "viewer"];
let activePollTimer = null;
let inspectionInProgress = false;
let cachedHistoryItems = [];
let activeHistoryFilter = "all";
let activeHistorySearch = "";

initSession();
syncAnalysisMode();
syncInspectionSettings();
loadAppConfig();

loginTab.addEventListener("click", () => showAuthForm("login"));
registerTab.addEventListener("click", () => showAuthForm("register"));
logoutButton.addEventListener("click", logout);
pdfReport.addEventListener("click", downloadPdfReport);
zipReport.addEventListener("click", downloadZipReport);
refreshHistoryButton.addEventListener("click", refreshInspectionHistory);
exportHistoryCsvButton.addEventListener("click", downloadHistoryCsv);
refreshSystemButton.addEventListener("click", refreshSystemStatus);
refreshUsersButton.addEventListener("click", refreshAdminUsers);
settingsForm.addEventListener("submit", handleSettingsSubmit);
downloadBackupButton.addEventListener("click", downloadSystemBackup);
historyBody.addEventListener("click", handleHistoryAction);
adminUsersBody.addEventListener("click", handleAdminUserAction);
adminUsersBody.addEventListener("change", handleAdminUserChange);
form.addEventListener("input", syncInspectionSettings);
historySearchInput.addEventListener("input", () => {
  activeHistorySearch = historySearchInput.value.trim().toLowerCase();
  renderHistory(cachedHistoryItems);
});
analysisModeInputs.forEach((input) => {
  input.addEventListener("change", () => {
    syncAnalysisMode();
    syncInspectionSettings();
  });
});
historyFilterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeHistoryFilter = button.dataset.filter || "all";
    historyFilterButtons.forEach((item) => item.classList.toggle("active", item === button));
    renderHistory(cachedHistoryItems);
  });
});

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setAuthBusy(loginButton, true, "Logging in...", "Login");

  try {
    const payload = formJson(loginForm);
    const session = await authRequest("/api/auth/login", payload);
    saveSession(session);
    setStatus("Login successful. You can run an inspection now.", false);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setAuthBusy(loginButton, false, "Logging in...", "Login");
  }
});

registerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setAuthBusy(registerButton, true, "Creating...", "Create Account");

  try {
    const payload = formJson(registerForm);
    const session = await authRequest("/api/auth/register", payload);
    saveSession(session);
    setStatus("Account created. You can run an inspection now.", false);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setAuthBusy(registerButton, false, "Creating...", "Create Account");
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (inspectionInProgress) {
    setStatus("An inspection is already processing. Please wait for it to finish.", true);
    return;
  }

  const token = getToken();
  if (!token) {
    setStatus("Please login before running an inspection.", true);
    return;
  }

  setStatus("Running inspection...", false);
  setBusy(true);
  results.hidden = true;
  jobPanel.hidden = true;
  setReportAvailability(false, "Reports will appear after the inspection completes.");
  clearPollTimer();

  const formData = new FormData(form);
  const analysisMode = getAnalysisMode();
  formData.delete("analysis_mode");
  if (analysisMode === "image") {
    formData.delete("video");
  } else {
    formData.delete("image");
  }
  if (!formData.get("model_path")) {
    formData.delete("model_path");
  }

  try {
    const response = await fetch("/api/inspections", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });
    const payload = await response.json();
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      throw new Error(payload.detail || "Inspection failed.");
    }

    renderJob(payload);
    setStatus("Inspection queued. Processing will continue in the background.", false);
    loadInspectionHistory();
    pollJob(payload.job_id);
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
    jobPanel.hidden = true;
    setBusy(false);
  }
});

function initSession() {
  const token = getToken();
  const user = getUser();
  if (token && user) {
    showAuthenticated(user);
  } else {
    showAnonymous();
  }
}

function showAuthForm(mode) {
  const isLogin = mode === "login";
  loginForm.hidden = !isLogin;
  registerForm.hidden = isLogin;
  loginTab.classList.toggle("active", isLogin);
  registerTab.classList.toggle("active", !isLogin);
}

async function authRequest(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Authentication failed.");
  }
  return data;
}

function saveSession(session) {
  localStorage.setItem(TOKEN_KEY, session.access_token);
  localStorage.setItem(USER_KEY, JSON.stringify(session.user));
  showAuthenticated(session.user);
}

function logout() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  clearPollTimer();
  jobPanel.hidden = true;
  results.hidden = true;
  historyPanel.hidden = true;
  dashboardPanel.hidden = true;
  systemPanel.hidden = true;
  settingsPanel.hidden = true;
  adminPanel.hidden = true;
  setBusy(false);
  setStatus("Logged out.", false);
  showAnonymous();
}

function expireSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  clearPollTimer();
  jobPanel.hidden = true;
  results.hidden = true;
  historyPanel.hidden = true;
  dashboardPanel.hidden = true;
  systemPanel.hidden = true;
  settingsPanel.hidden = true;
  adminPanel.hidden = true;
  setBusy(false);
  showAnonymous();
}

function showAuthenticated(user) {
  authPanel.hidden = true;
  dashboardPanel.hidden = false;
  inspectionPanel.hidden = false;
  historyPanel.hidden = false;
  systemPanel.hidden = false;
  settingsPanel.hidden = user.role !== "admin";
  adminPanel.hidden = user.role !== "admin";
  sessionBox.hidden = false;
  sessionUser.textContent = user.full_name || user.email;
  loadInspectionHistory();
  loadSystemStatus();
  if (user.role === "admin") {
    loadSettings();
    loadAdminUsers();
  }
}

function showAnonymous() {
  authPanel.hidden = false;
  dashboardPanel.hidden = true;
  inspectionPanel.hidden = true;
  historyPanel.hidden = true;
  systemPanel.hidden = true;
  settingsPanel.hidden = true;
  adminPanel.hidden = true;
  sessionBox.hidden = true;
}

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function getUser() {
  const rawUser = localStorage.getItem(USER_KEY);
  if (!rawUser) {
    return null;
  }

  try {
    return JSON.parse(rawUser);
  } catch {
    localStorage.removeItem(USER_KEY);
    return null;
  }
}

function formJson(targetForm) {
  return Object.fromEntries(new FormData(targetForm).entries());
}

function setAuthBusy(button, isBusy, busyLabel, idleLabel) {
  button.disabled = isBusy;
  button.textContent = isBusy ? busyLabel : idleLabel;
}

function setBusy(isBusy) {
  inspectionInProgress = isBusy;
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "Processing..." : "Run Inspection";
}

function getAnalysisMode() {
  const selected = document.querySelector("input[name='analysis_mode']:checked");
  return selected?.value === "image" ? "image" : "video";
}

function syncAnalysisMode() {
  const isImageMode = getAnalysisMode() === "image";
  videoUploadField.hidden = isImageMode;
  imageUploadField.hidden = !isImageMode;
  videoInput.required = !isImageMode;
  imageInput.required = isImageMode;
  if (isImageMode) {
    videoInput.value = "";
    return;
  }
  imageInput.value = "";
}

function setStatus(message, isError) {
  statusBox.hidden = false;
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function renderJob(job) {
  const progress = Number(job.progress_percent ?? 0);
  jobPanel.hidden = false;
  jobIdBox.textContent = job.job_id || "-";
  jobStatusBox.textContent = job.status || "-";
  jobProgressBox.textContent = `${progress}%`;
  jobProgressBar.style.width = `${Math.min(Math.max(progress, 0), 100)}%`;
  jobResultState.textContent = job.status === "completed" ? "Ready" : "Waiting";
  jobStageLabel.textContent = formatStageLabel(job.status, progress);
  updateWorkflow(progress, job.status);
}

async function pollJob(jobId) {
  const token = getToken();
  if (!token) {
    return;
  }

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const payload = await response.json();
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to fetch job status.");
    }

    renderJob(payload);

    if (payload.status === "completed") {
      clearPollTimer();
      await fetchJobResult(jobId);
      loadInspectionHistory();
      setBusy(false);
      return;
    }

    if (payload.status === "failed") {
      clearPollTimer();
      jobResultState.textContent = "Failed";
      setStatus(toFriendlyError(payload.error_message || "Inspection failed."), true);
      loadInspectionHistory();
      setBusy(false);
      return;
    }

    activePollTimer = window.setTimeout(() => pollJob(jobId), POLL_INTERVAL_MS);
  } catch (error) {
    clearPollTimer();
    setStatus(toFriendlyError(error.message), true);
    setBusy(false);
  }
}

async function fetchJobResult(jobId) {
  const token = getToken();
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/result`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  const payload = await response.json();
  if (response.status === 401) {
    expireSession();
    throw new Error("Your login session expired. Please login again.");
  }
  if (!response.ok) {
    throw new Error(payload.detail || "Unable to fetch inspection result.");
  }

  renderResults(payload);
  jobResultState.textContent = "Ready";
  setStatus("Inspection complete.", false);
}

function clearPollTimer() {
  if (activePollTimer !== null) {
    window.clearTimeout(activePollTimer);
    activePollTimer = null;
  }
}

function renderResults(payload) {
  const summary = payload.summary;
  const classCounts = summary.unique_detections_by_class || summary.detections_by_class || {};
  const topClasses = formatTopClasses(classCounts);

  summaryGrid.innerHTML = "";
  addMetric("Panels Needing Attention", summary.defect_areas_needing_attention ?? summary.estimated_unique_detections ?? summary.total_detections);
  addMetric("Immediate Safety Review", summary.critical_defects ?? 0);
  addMetric("Repair Priority", summary.high_defects ?? 0);
  addMetric("Planning Reliability", summary.report_reliability || "Review");
  addMetric("Frames Reviewed", summary.total_frames_sampled);
  addMetric("Most Observed Defect", topClasses);
  renderInspectionMetadata(summary.video_metadata?.inspection_metadata || {});
  renderPriorityList(summary.maintenance_priority_list || []);
  updateDashboardFromSummary(summary);

  jsonReport.href = payload.report.json_url || "#";
  htmlReport.href = payload.report.html_url || "#";
  pdfReport.dataset.jobId = payload.job_id;
  zipReport.dataset.jobId = payload.job_id;
  setReportAvailability(
    Boolean(payload.report.json_url && payload.report.html_url && payload.job_id),
    "JSON, HTML, PDF, and ZIP report package are ready."
  );

  framesContainer.innerHTML = "";
  payload.frames.forEach((frame, index) => {
    framesContainer.appendChild(renderFrame(frame, index));
  });

  results.hidden = false;
}

async function downloadPdfReport(event) {
  event.preventDefault();
  const token = getToken();
  const jobId = pdfReport.dataset.jobId;
  if (!token || !jobId) {
    setStatus("PDF report is not ready yet.", true);
    return;
  }

  await downloadPdfForJob(jobId);
}

async function downloadZipReport(event) {
  event.preventDefault();
  const token = getToken();
  const jobId = zipReport.dataset.jobId;
  if (!token || !jobId) {
    setStatus("ZIP report package is not ready yet.", true);
    return;
  }

  await downloadReportPackageForJob(jobId);
}

async function refreshInspectionHistory(event) {
  event.preventDefault();
  await loadInspectionHistory({ notify: true });
}

async function refreshSystemStatus(event) {
  event.preventDefault();
  await loadSystemStatus({ notify: true });
}

async function loadSystemStatus(options = {}) {
  const token = getToken();
  if (!token) {
    return;
  }

  const shouldNotify = Boolean(options.notify);
  if (shouldNotify) {
    refreshSystemButton.disabled = true;
    refreshSystemButton.textContent = "Refreshing...";
  }

  try {
    const response = await fetch("/api/system/status", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const payload = await response.json();
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to load system status.");
    }
    renderSystemStatus(payload);
    if (shouldNotify) {
      setStatus("System status refreshed.", false);
    }
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
  } finally {
    if (shouldNotify) {
      refreshSystemButton.disabled = false;
      refreshSystemButton.textContent = "Refresh Status";
    }
  }
}

function renderSystemStatus(status) {
  systemStatusGrid.innerHTML = "";
  const values = [
    ["App", `${status.app?.name || "SkyVision-DL"} ${status.app?.version || ""}`],
    ["Database", status.database?.status || "-"],
    ["Model", status.model?.exists ? "Available" : "Missing"],
    ["Model Path", status.model?.path || "-"],
    ["Storage Free", `${status.storage?.free_mb ?? "-"} MB`],
    ["Upload Limits", `Video ${status.uploads?.max_video_mb ?? "-"} MB / Image ${status.uploads?.max_image_mb ?? "-"} MB`],
  ];
  values.forEach(([label, value]) => addSystemMetric(label, value));
}

function addSystemMetric(label, value) {
  const metric = document.createElement("div");
  metric.className = "metric";
  metric.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong>`;
  systemStatusGrid.appendChild(metric);
}

async function loadSettings() {
  const token = getToken();
  if (!token) {
    return;
  }

  try {
    const response = await fetch("/api/settings", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const payload = await response.json();
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to load settings.");
    }
    applyLocalSettings(payload.settings || {});
  } catch (error) {
    settingsStatus.textContent = toFriendlyError(error.message);
  }
}

async function handleSettingsSubmit(event) {
  event.preventDefault();
  const token = getToken();
  if (!token) {
    setStatus("Please login before updating settings.", true);
    return;
  }

  saveSettingsButton.disabled = true;
  saveSettingsButton.textContent = "Saving...";
  settingsStatus.textContent = "Saving local configuration...";

  const payload = {
    default_sample_fps: Number(settingsInputs.sampleFps.value),
    default_confidence: Number(settingsInputs.confidence.value),
    default_device: settingsInputs.device.value.trim(),
    default_model_path: settingsInputs.modelPath.value.trim(),
    max_video_upload_mb: Number(settingsInputs.maxVideoUploadMb.value),
    max_image_upload_mb: Number(settingsInputs.maxImageUploadMb.value),
  };

  try {
    const response = await fetch("/api/settings", {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      throw new Error(data.detail || "Unable to save settings.");
    }
    applyLocalSettings(data.settings || {});
    await loadSystemStatus();
    setStatus("Application settings saved.", false);
    settingsStatus.textContent = "Local defaults updated for new inspections.";
  } catch (error) {
    const friendlyMessage = toFriendlyError(error.message);
    setStatus(friendlyMessage, true);
    settingsStatus.textContent = friendlyMessage;
  } finally {
    saveSettingsButton.disabled = false;
    saveSettingsButton.textContent = "Save Settings";
  }
}

function applyLocalSettings(settings) {
  if (!settings) {
    return;
  }

  const sampleFps = settings.default_sample_fps ?? "1.0";
  const confidence = settings.default_confidence ?? "0.35";
  const device = settings.default_device ?? "0";
  const modelPath = settings.default_model_path ?? "models/weights/best.pt";

  document.querySelector("#sample_fps").value = sampleFps;
  document.querySelector("#confidence").value = confidence;
  document.querySelector("#device").value = device;
  document.querySelector("#model_path").value = modelPath;

  settingsInputs.sampleFps.value = sampleFps;
  settingsInputs.confidence.value = confidence;
  settingsInputs.device.value = device;
  settingsInputs.modelPath.value = modelPath;
  settingsInputs.maxVideoUploadMb.value = settings.max_video_upload_mb ?? 500;
  settingsInputs.maxImageUploadMb.value = settings.max_image_upload_mb ?? 50;
  syncInspectionSettings();
}

async function downloadHistoryCsv() {
  const token = getToken();
  if (!token) {
    setStatus("Please login before exporting inspection history.", true);
    return;
  }

  exportHistoryCsvButton.disabled = true;
  exportHistoryCsvButton.textContent = "Exporting...";
  try {
    await downloadBlob("/api/inspections/history.csv", "skyvision_inspection_history.csv", token);
    setStatus("Inspection history CSV exported.", false);
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
  } finally {
    exportHistoryCsvButton.disabled = false;
    exportHistoryCsvButton.textContent = "Export CSV";
  }
}

async function downloadSystemBackup() {
  const token = getToken();
  if (!token) {
    setStatus("Please login before downloading a backup.", true);
    return;
  }

  downloadBackupButton.disabled = true;
  downloadBackupButton.textContent = "Preparing...";
  try {
    await downloadBlob("/api/system/backup.zip", "skyvision_backup.zip", token);
    setStatus("Local backup package downloaded.", false);
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
  } finally {
    downloadBackupButton.disabled = false;
    downloadBackupButton.textContent = "Download Backup";
  }
}

async function downloadBlob(url, fallbackFilename, token) {
  const response = await fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (response.status === 401) {
    expireSession();
    throw new Error("Your login session expired. Please login again.");
  }
  if (!response.ok) {
    const payload = await response.json();
    throw new Error(payload.detail || "Unable to download file.");
  }

  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filenameFromDisposition(response.headers.get("Content-Disposition")) || fallbackFilename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}

function filenameFromDisposition(disposition) {
  if (!disposition) {
    return "";
  }
  const match = disposition.match(/filename="?([^"]+)"?/i);
  return match ? match[1] : "";
}

async function refreshAdminUsers(event) {
  event.preventDefault();
  await loadAdminUsers({ notify: true });
}

async function loadAdminUsers(options = {}) {
  const token = getToken();
  const user = getUser();
  if (!token || user?.role !== "admin") {
    return;
  }

  const shouldNotify = Boolean(options.notify);
  if (shouldNotify) {
    refreshUsersButton.disabled = true;
    refreshUsersButton.textContent = "Refreshing...";
  }

  try {
    const response = await fetch("/api/auth/admin/users", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const payload = await response.json();
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (response.status === 403) {
      adminPanel.hidden = true;
      return;
    }
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to load users.");
    }
    renderAdminUsers(payload || []);
    if (shouldNotify) {
      setStatus("User list refreshed.", false);
    }
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
  } finally {
    if (shouldNotify) {
      refreshUsersButton.disabled = false;
      refreshUsersButton.textContent = "Refresh Users";
    }
  }
}

function renderAdminUsers(users) {
  adminUsersBody.innerHTML = "";
  adminEmpty.hidden = users.length > 0;
  adminEmpty.textContent = users.length ? "" : "No users found.";

  users.forEach((user) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(user.full_name || "-")}</td>
      <td>${escapeHtml(user.email || "-")}</td>
      <td>
        <select class="admin-select" data-user-id="${escapeHtml(String(user.id))}" data-field="role">
          ${USER_ROLES.map((role) => `
            <option value="${escapeHtml(role)}" ${role === user.role ? "selected" : ""}>${escapeHtml(formatRoleName(role))}</option>
          `).join("")}
        </select>
      </td>
      <td><span class="status-pill ${user.is_active ? "completed" : "failed"}">${user.is_active ? "Active" : "Inactive"}</span></td>
      <td>
        <button class="secondary-button small-button" type="button" data-action="toggle-active" data-user-id="${escapeHtml(String(user.id))}" data-active="${user.is_active ? "1" : "0"}">
          ${user.is_active ? "Deactivate" : "Activate"}
        </button>
      </td>
    `;
    adminUsersBody.appendChild(row);
  });
}

async function handleAdminUserChange(event) {
  const select = event.target.closest("select[data-field='role']");
  if (!select) {
    return;
  }
  await updateAdminUser(select.dataset.userId, { role: select.value });
}

async function handleAdminUserAction(event) {
  const button = event.target.closest("button[data-action='toggle-active']");
  if (!button) {
    return;
  }
  const isActive = button.dataset.active === "1";
  await updateAdminUser(button.dataset.userId, { is_active: !isActive });
}

async function updateAdminUser(userId, payload) {
  const token = getToken();
  if (!token || !userId) {
    return;
  }

  try {
    const response = await fetch(`/api/auth/admin/users/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      throw new Error(data.detail || "Unable to update user.");
    }
    await loadAdminUsers();
    setStatus("User updated.", false);
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
    await loadAdminUsers();
  }
}

async function loadInspectionHistory(options = {}) {
  const token = getToken();
  if (!token) {
    return;
  }

  const shouldNotify = Boolean(options.notify);
  if (shouldNotify) {
    refreshHistoryButton.disabled = true;
    refreshHistoryButton.textContent = "Refreshing...";
  }

  try {
    const response = await fetch("/api/inspections/history", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const payload = await response.json();
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to load inspection history.");
    }

    cachedHistoryItems = payload.items || [];
    renderHistory(cachedHistoryItems);
    updateDashboardMetrics(cachedHistoryItems);
    if (shouldNotify) {
      setStatus("Inspection history refreshed.", false);
    }
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
  } finally {
    if (shouldNotify) {
      refreshHistoryButton.disabled = false;
      refreshHistoryButton.textContent = "Refresh";
    }
  }
}

function renderHistory(items) {
  historyBody.innerHTML = "";
  const filteredItems = filterHistoryItems(items);
  historyEmpty.hidden = filteredItems.length > 0;
  historyEmpty.textContent = items.length
    ? "No inspections match the selected filter."
    : "No inspections yet. Run a video or image inspection to build the maintenance history.";

  filteredItems.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><code>${escapeHtml(item.job_id)}</code></td>
      <td>${escapeHtml(item.video_filename || "-")}</td>
      <td><span class="status-pill ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span></td>
      <td>${escapeHtml(formatDateTime(item.created_at))}</td>
      <td>${escapeHtml(formatDateTime(item.completed_at))}</td>
      <td>${escapeHtml(formatAttentionAreas(item))}</td>
      <td>
        <div class="row-actions">
          <button class="secondary-button small-button" type="button" data-action="open" data-job-id="${escapeHtml(item.job_id)}" ${item.status === "completed" ? "" : "disabled"}>Open</button>
          <button class="secondary-button small-button" type="button" data-action="pdf" data-job-id="${escapeHtml(item.job_id)}" ${item.status === "completed" ? "" : "disabled"}>PDF</button>
          <button class="secondary-button small-button" type="button" data-action="zip" data-job-id="${escapeHtml(item.job_id)}" ${item.status === "completed" ? "" : "disabled"}>ZIP</button>
          <button class="secondary-button small-button" type="button" data-action="${item.archived ? "restore" : "archive"}" data-job-id="${escapeHtml(item.job_id)}" ${["completed", "failed"].includes(item.status) ? "" : "disabled"}>${item.archived ? "Restore" : "Archive"}</button>
        </div>
      </td>
    `;
    historyBody.appendChild(row);
  });
}

function filterHistoryItems(items) {
  const filtered = items.filter((item) => {
    if (activeHistoryFilter !== "archived" && item.archived) {
      return false;
    }
    if (activeHistoryFilter === "archived") {
      return Boolean(item.archived);
    }
    if (activeHistoryFilter === "completed" && item.status !== "completed") {
      return false;
    }
    if (activeHistoryFilter === "failed" && item.status !== "failed") {
      return false;
    }
    if (
      activeHistoryFilter === "priority" &&
      Number(item.critical_defects ?? 0) + Number(item.high_defects ?? 0) <= 0
    ) {
      return false;
    }
    return true;
  });

  if (!activeHistorySearch) {
    return filtered;
  }

  return filtered.filter((item) => {
    const searchable = [
      item.job_id,
      item.video_filename,
      item.status,
      formatDateTime(item.created_at),
      formatDateTime(item.completed_at),
      formatAttentionAreas(item),
    ]
      .join(" ")
      .toLowerCase();
    return searchable.includes(activeHistorySearch);
  });
}

function updateDashboardMetrics(items) {
  const completed = items.filter((item) => item.status === "completed");
  const reportsReady = completed.filter((item) => item.pdf_report_url || item.report?.pdf_url);
  const uniqueTotal = completed.reduce((sum, item) => {
    const unique = item.estimated_unique_detections;
    return sum + Number(item.defect_areas_needing_attention ?? unique ?? item.total_detections ?? 0);
  }, 0);
  const criticalTotal = completed.reduce((sum, item) => sum + Number(item.critical_defects ?? 0), 0);
  const highTotal = completed.reduce((sum, item) => sum + Number(item.high_defects ?? 0), 0);
  const latestCompleted = completed[0];

  dashboardMetrics.total.textContent = String(items.length);
  dashboardMetrics.completed.textContent = String(completed.length);
  dashboardMetrics.uniqueDefects.textContent = String(uniqueTotal);
  dashboardMetrics.reportsReady.textContent = String(reportsReady.length);
  dashboardMetrics.critical.textContent = String(criticalTotal);
  dashboardMetrics.high.textContent = String(highTotal);
  dashboardMetrics.reliability.textContent = latestCompleted?.report_reliability || "-";
  dashboardMetrics.primaryDefect.textContent = "-";
  updateDashboardRisk(criticalTotal, highTotal);
}

function updateWorkflow(progress, status) {
  const completed = status === "completed";
  workflowSteps.extract.classList.toggle("active", progress >= 10);
  workflowSteps.yolo.classList.toggle("active", progress >= 35);
  workflowSteps.results.classList.toggle("active", progress >= 90 || completed);
  workflowSteps.report.classList.toggle("active", completed);
  workflowCurrentState.textContent = formatStageLabel(status, progress);
}

function formatStageLabel(status, progress) {
  if (status === "completed") {
    return "Maintenance Report Ready";
  }
  if (status === "failed") {
    return "Failed";
  }
  if (progress >= 90) {
    return "Finalizing";
  }
  if (progress >= 35) {
    return "Detecting";
  }
  if (progress >= 10) {
    return "Extracting";
  }
  return status || "Waiting";
}

function formatAttentionAreas(item) {
  const areas = item.defect_areas_needing_attention ?? item.estimated_unique_detections;
  if (areas === null || areas === undefined) {
    return "-";
  }
  const serious = Number(item.critical_defects ?? 0) + Number(item.high_defects ?? 0);
  if (serious > 0) {
    return `${areas} area(s), ${serious} priority`;
  }
  return `${areas} area(s)`;
}

async function handleHistoryAction(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) {
    return;
  }

  const jobId = button.dataset.jobId;
  if (!jobId) {
    return;
  }

  if (button.dataset.action === "open") {
    await openHistoryResult(jobId);
  }

  if (button.dataset.action === "pdf") {
    await downloadPdfForJob(jobId);
  }

  if (button.dataset.action === "zip") {
    await downloadReportPackageForJob(jobId);
  }

  if (button.dataset.action === "archive") {
    await setInspectionArchived(jobId, true);
  }

  if (button.dataset.action === "restore") {
    await setInspectionArchived(jobId, false);
  }
}

async function openHistoryResult(jobId) {
  try {
    renderJob({
      job_id: jobId,
      status: "completed",
      progress_percent: 100,
    });
    await fetchJobResult(jobId);
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
  }
}

async function downloadPdfForJob(jobId) {
  const token = getToken();
  if (!token) {
    setStatus("Please login before downloading a PDF report.", true);
    return;
  }

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/report/pdf`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Unable to download PDF report.");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `skyvision_${jobId}_inspection_report.pdf`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
  }
}

async function downloadReportPackageForJob(jobId) {
  const token = getToken();
  if (!token) {
    setStatus("Please login before downloading a report package.", true);
    return;
  }

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/report/package.zip`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Unable to download report package.");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `skyvision_${jobId}_report_package.zip`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
  }
}

async function setInspectionArchived(jobId, archived) {
  const token = getToken();
  if (!token) {
    setStatus("Please login before updating inspection history.", true);
    return;
  }

  try {
    const response = await fetch(`/api/inspections/${encodeURIComponent(jobId)}/archive`, {
      method: "PATCH",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ archived }),
    });
    const payload = await response.json();
    if (response.status === 401) {
      expireSession();
      throw new Error("Your login session expired. Please login again.");
    }
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to update inspection archive state.");
    }
    await loadInspectionHistory();
    setStatus(archived ? "Inspection archived." : "Inspection restored.", false);
  } catch (error) {
    setStatus(toFriendlyError(error.message), true);
  }
}

function setReportAvailability(isAvailable, message) {
  [jsonReport, htmlReport, pdfReport, zipReport].forEach((link) => {
    link.classList.toggle("unavailable", !isAvailable);
    link.setAttribute("aria-disabled", String(!isAvailable));
  });
  reportStatus.textContent = message;
}

function syncInspectionSettings() {
  const formData = new FormData(form);
  const analysisMode = getAnalysisMode();
  const sampleFps = formData.get("sample_fps") || "1.0";
  const confidence = formData.get("confidence") || "0.35";
  const device = formData.get("device") || "0";
  const modelPath = formData.get("model_path") || "models/weights/best.pt";

  dashboardSettings.sampleFps.textContent = analysisMode === "image" ? "Single image" : sampleFps;
  dashboardSettings.confidence.textContent = confidence;
  dashboardSettings.device.textContent = device;
  dashboardSettings.modelPath.textContent = modelPath;
  dashboardSettings.productModel.textContent = `Model: ${modelPath}`;
  dashboardSettings.productConfidence.textContent = `Confidence: ${confidence}`;
  dashboardSettings.productDevice.textContent = `Device: ${device}`;
}

async function loadAppConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) {
      return;
    }
    const config = await response.json();
    if (config.settings) {
      applyLocalSettings(config.settings);
    }
    if (config.default_model_path) {
      document.querySelector("#model_path").value = config.default_model_path;
      if (settingsInputs.modelPath) {
        settingsInputs.modelPath.value = config.default_model_path;
      }
    }
    if (config.model?.version_label) {
      dashboardSettings.modelVersion.textContent = config.model.version_label;
    }
    syncInspectionSettings();
  } catch {
    dashboardSettings.modelVersion.textContent = "Unavailable";
  }
}

function toFriendlyError(message) {
  if (!message) {
    return "Something went wrong. Please try again.";
  }

  if (message.includes("Failed to fetch")) {
    return "Cannot reach the local server. Please restart SkyVision-DL and refresh the page.";
  }

  if (message.includes("YOLO inference")) {
    return "YOLO could not process the sampled frame. Try model path 'yolov8n.pt' and device 'cpu', then run the inspection again.";
  }

  if (message.includes("model weights not found")) {
    return "Model weights were not found. Use a valid local .pt file or the standard 'yolov8n.pt' fallback.";
  }

  if (message.includes("Could not validate credentials") || message.includes("login session expired")) {
    return "Your login session expired. Please login again.";
  }

  return message;
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "-";
  }

  return parsed.toLocaleString();
}

function addMetric(label, value) {
  const metric = document.createElement("div");
  metric.className = "metric";
  metric.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong>`;
  summaryGrid.appendChild(metric);
}

function renderInspectionMetadata(metadata) {
  if (!metadata || !Object.keys(metadata).length) {
    return;
  }
  const labels = {
    organization_name: "Organization",
    project_name: "Project",
    inspector_name: "Inspector",
    building_name: "Building",
    floor_level: "Floor / Level",
    facade_side: "Facade Side",
    location: "Location",
    inspection_date: "Inspection Date",
  };
  Object.entries(labels).forEach(([key, label]) => {
    if (metadata[key]) {
      addMetric(label, metadata[key]);
    }
  });
}

function updateDashboardFromSummary(summary) {
  const critical = Number(summary.critical_defects ?? 0);
  const high = Number(summary.high_defects ?? 0);
  dashboardMetrics.uniqueDefects.textContent = String(
    summary.defect_areas_needing_attention ?? summary.estimated_unique_detections ?? summary.total_detections ?? 0
  );
  dashboardMetrics.critical.textContent = String(critical);
  dashboardMetrics.high.textContent = String(high);
  dashboardMetrics.reliability.textContent = summary.report_reliability || "Review";
  dashboardMetrics.primaryDefect.textContent = formatPrimaryDefect(
    summary.unique_detections_by_class || summary.detections_by_class || {}
  );
  updateDashboardRisk(critical, high);
}

function updateDashboardRisk(critical, high) {
  dashboardRiskLabel.classList.remove("risk-low", "risk-high", "risk-critical");
  if (critical > 0) {
    dashboardRiskLabel.textContent = `${critical} immediate safety review area(s)`;
    dashboardRiskLabel.classList.add("risk-critical");
    return;
  }
  if (high > 0) {
    dashboardRiskLabel.textContent = `${high} repair-priority area(s)`;
    dashboardRiskLabel.classList.add("risk-high");
    return;
  }
  dashboardRiskLabel.textContent = "No urgent defects";
  dashboardRiskLabel.classList.add("risk-low");
}

function formatTopClasses(classCounts) {
  const entries = Object.entries(classCounts);
  if (!entries.length) {
    return "None";
  }
  return entries
    .sort((first, second) => Number(second[1]) - Number(first[1]))
    .map(([name, count]) => `${formatClassName(name)}: ${count}`)
    .join(", ");
}

function formatPrimaryDefect(classCounts) {
  const [top] = Object.entries(classCounts).sort((first, second) => Number(second[1]) - Number(first[1]));
  if (!top) {
    return "-";
  }
  return `${formatClassName(top[0])} (${top[1]})`;
}

function renderPriorityList(items) {
  priorityList.innerHTML = "";
  if (!items.length) {
    priorityList.innerHTML = `
      <div class="priority-empty">
        No maintenance action areas were detected in this inspection.
      </div>
    `;
    return;
  }

  const heading = document.createElement("div");
  heading.className = "priority-heading";
  heading.innerHTML = `
    <h3>Maintenance Priority</h3>
    <p>Repeated video-frame observations are grouped into panel-level repair priorities.</p>
  `;
  priorityList.appendChild(heading);

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = `priority-card severity-${String(item.severity || "medium").toLowerCase()}`;
    card.innerHTML = `
      <div class="priority-card-header">
        <span>Priority ${escapeHtml(String(item.priority || "-"))}</span>
        <strong>${escapeHtml(formatClassName(item.class_name))}</strong>
        <em>${escapeHtml(item.severity || "Review")}</em>
      </div>
      <dl>
        <div><dt>Location</dt><dd>${escapeHtml(titleCase(item.approximate_location || "Facade area"))}</dd></div>
        <div><dt>Evidence</dt><dd>${escapeHtml(String(item.evidence_frame_count || 0))} frame(s)</dd></div>
        <div><dt>Confidence</dt><dd>${Number(item.confidence || 0).toFixed(2)}</dd></div>
        <div><dt>Reliability</dt><dd>${escapeHtml(item.reliability || "Review")}</dd></div>
      </dl>
      <p>${escapeHtml(item.recommended_action || "Review this facade area during maintenance inspection.")}</p>
    `;
    priorityList.appendChild(card);
  });
}

function formatClassName(value) {
  return titleCase(String(value || "Defect").replaceAll("_", " "));
}

function titleCase(value) {
  return String(value)
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatRoleName(value) {
  return titleCase(String(value || "").replaceAll("_", " "));
}

function renderFrame(frame, index) {
  const card = document.createElement("article");
  card.className = "frame-card";
  const detectionsList = frame.detections || [];
  const defectCount = detectionsList.length;
  const highestSeverity = highestFrameSeverity(detectionsList);
  const topDefect = topFrameDefect(detectionsList);

  const detections = defectCount
    ? detectionsList
      .map((item) => `<li>${escapeHtml(item.class_name)} ${Number(item.confidence).toFixed(2)}</li>`)
      .join("")
    : "<li>No defects</li>";

  const heatmapMarkup = frame.heatmap_image_url
    ? `
      <figure>
        <img src="${frame.heatmap_image_url}" alt="Defect heatmap frame ${index + 1}">
        <figcaption>Defect attention heatmap</figcaption>
      </figure>
    `
    : "";

  card.innerHTML = `
    <div class="frame-visuals">
      <figure>
        <img src="${frame.annotated_image_url}" alt="Annotated frame ${index + 1}">
        <figcaption>Annotated defect view</figcaption>
      </figure>
      ${heatmapMarkup}
    </div>
    <div class="frame-body">
      <div class="frame-card-header">
        <div>
          <span class="frame-kicker">Frame ${index + 1}</span>
          <h3>${escapeHtml(topDefect)}</h3>
        </div>
        <span class="frame-severity severity-${highestSeverity.toLowerCase()}">${highestSeverity}</span>
      </div>
      <div class="frame-meta">
        <span>${defectCount} defect observation(s)</span>
        <span>${defectCount ? "Review highlighted panel areas" : "No action from this frame"}</span>
      </div>
      <ul class="detections">${detections}</ul>
    </div>
  `;

  return card;
}

function topFrameDefect(detections) {
  if (!detections.length) {
    return "No Defects Observed";
  }
  const counts = detections.reduce((accumulator, item) => {
    const name = formatClassName(item.class_name);
    accumulator[name] = (accumulator[name] || 0) + 1;
    return accumulator;
  }, {});
  const [top] = Object.entries(counts).sort((first, second) => second[1] - first[1]);
  return `${top[0]} (${top[1]})`;
}

function highestFrameSeverity(detections) {
  if (!detections.length) {
    return "Clear";
  }
  const ranked = detections
    .map((item) => severityForDetection(item))
    .sort((first, second) => severityRank(second) - severityRank(first));
  return ranked[0];
}

function severityForDetection(detection) {
  const name = String(detection.class_name || "").toLowerCase().replaceAll(" ", "_");
  const confidence = Number(detection.confidence || 0);
  if (name === "broken_glass") {
    return confidence >= 0.70 ? "Critical" : "High";
  }
  if (name === "fracture") {
    return confidence >= 0.55 ? "Critical" : "High";
  }
  if (name === "crack") {
    return confidence >= 0.55 ? "High" : "Medium";
  }
  if (name === "water_leakage" || name === "delamination") {
    return confidence >= 0.45 ? "High" : "Medium";
  }
  if (name === "surface_stain" || name === "repair_patch") {
    return confidence >= 0.45 ? "Medium" : "Low";
  }
  return "Medium";
}

function severityRank(severity) {
  return { Critical: 4, High: 3, Medium: 2, Low: 1, Clear: 0 }[severity] || 0;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
