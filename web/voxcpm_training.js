import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const TRAINING_WIDGET_STYLES = `
  .voxcpm-training-widget {
    --primary: #06b6d4;
    --primary-glow: rgba(6, 182, 212, 0.4);
    --secondary: #8b5cf6;
    --success: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;
    --bg-dark: #0f0f12;
    --bg-card: #18181b;
    --bg-elevated: #1f1f23;
    --border: #27272a;
    --border-hover: #3f3f46;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;

    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    box-sizing: border-box;
    height: 100%;
    min-height: 300px;
    display: flex;
    flex-direction: column;
  }

  .voxcpm-training-widget * {
    box-sizing: border-box;
  }

  /* Header */
  .voxcpm-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .voxcpm-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .voxcpm-badge {
    padding: 2px 8px;
    background: var(--primary);
    color: white;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
  }

  .voxcpm-badge.idle {
    background: var(--text-muted);
  }

  .voxcpm-badge.training {
    background: var(--success);
    animation: voxcpm-pulse 2s infinite;
  }

  @keyframes voxcpm-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  /* Content */
  .voxcpm-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 10px;
    gap: 8px;
    overflow: hidden;
  }

  /* Stats */
  .voxcpm-stats {
    display: flex;
    gap: 12px;
    justify-content: center;
    align-items: baseline;
    flex-shrink: 0;
  }

  .voxcpm-stat {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .voxcpm-stat-label {
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .voxcpm-stat-value {
    font-size: 11px;
    font-weight: 600;
    color: var(--primary);
    font-variant-numeric: tabular-nums;
  }

  /* Progress Bar */
  .voxcpm-progress-section {
    background: var(--bg-elevated);
    border-radius: 6px;
    padding: 8px 10px;
    flex-shrink: 0;
  }

  .voxcpm-progress-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
  }

  .voxcpm-progress-label {
    font-size: 9px;
    color: var(--text-secondary);
  }

  .voxcpm-progress-value {
    font-size: 9px;
    color: var(--text-primary);
    font-weight: 500;
  }

  .voxcpm-progress-bar {
    height: 4px;
    background: var(--bg-dark);
    border-radius: 2px;
    overflow: hidden;
  }

  .voxcpm-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: 2px;
    transition: width 0.3s ease;
    width: 0%;
  }

  /* Loss Chart */
  .voxcpm-chart-section {
    background: var(--bg-elevated);
    border-radius: 6px;
    padding: 8px 10px;
    flex: 1;
    min-height: 80px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .voxcpm-chart-header {
    font-size: 9px;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  .voxcpm-chart-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }

  /* Status */
  .voxcpm-status {
    background: var(--bg-elevated);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10px;
    color: var(--text-secondary);
    text-align: center;
    border-left: 3px solid var(--primary);
    flex-shrink: 0;
  }

  .voxcpm-status.error {
    border-left-color: var(--danger);
    color: var(--danger);
  }

  .voxcpm-status.success {
    border-left-color: var(--success);
    color: var(--success);
  }

  /* Validation Audio Carousel */
  .voxcpm-preview-section {
    background: var(--bg-elevated);
    border-radius: 6px;
    padding: 8px 10px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .voxcpm-preview-header {
    font-size: 9px;
    color: var(--text-secondary);
    margin-bottom: 6px;
    flex-shrink: 0;
  }

  .voxcpm-preview-carousel {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    overflow-y: hidden;
    align-items: center;
    padding-bottom: 4px;
    scrollbar-width: thin;
    scrollbar-color: var(--border-hover) transparent;
  }

  .voxcpm-preview-carousel::-webkit-scrollbar {
    height: 4px;
  }
  .voxcpm-preview-carousel::-webkit-scrollbar-track {
    background: transparent;
  }
  .voxcpm-preview-carousel::-webkit-scrollbar-thumb {
    background: var(--border-hover);
    border-radius: 2px;
  }

  .voxcpm-preview-empty {
    width: 100%;
    text-align: center;
    font-size: 10px;
    color: var(--text-muted);
  }

  .voxcpm-preview-tile {
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
  }

  .voxcpm-play-btn {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 2px solid var(--border);
    background: var(--bg-dark);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    color: var(--text-secondary);
    font-size: 14px;
  }

  .voxcpm-play-btn:hover {
    border-color: var(--primary);
    color: var(--primary);
  }

  .voxcpm-play-btn.playing {
    border-color: var(--success);
    color: var(--success);
    animation: voxcpm-pulse 1.5s infinite;
  }

  .voxcpm-preview-tile .tile-label {
    font-size: 9px;
    color: var(--text-muted);
    text-align: center;
    white-space: nowrap;
  }
`;

// ---------------------------------------------------------------------------
// TrainingWidget
// ---------------------------------------------------------------------------

class TrainingWidget {
  constructor(options) {
    this.node = options.node;
    this.container = options.container;
    this.element = document.createElement("div");
    this.element.className = "voxcpm-training-widget";

    this.badgeEl = null;
    this.stepValueEl = null;
    this.lossValueEl = null;
    this.lrValueEl = null;
    this.progressFillEl = null;
    this.progressLabelEl = null;
    this.statusEl = null;
    this.canvasEl = null;

    this.lossHistory = [];
    this.isTraining = false;

    this.resizeObserver = null;
    this.resizeTimeout = null;

    this.injectStyles();
    this.createUI();
    this.container.appendChild(this.element);
  }

  injectStyles() {
    const styleId = "voxcpm-training-styles";
    if (!document.getElementById(styleId)) {
      const style = document.createElement("style");
      style.id = styleId;
      style.textContent = TRAINING_WIDGET_STYLES;
      document.head.appendChild(style);
    }
  }

  createUI() {
    this.element.innerHTML = `
      <div class="voxcpm-header">
        <div class="voxcpm-title">
          <span>VoxCPM Training</span>
          <span class="voxcpm-badge idle">Idle</span>
        </div>
      </div>

      <div class="voxcpm-content">
        <div class="voxcpm-stats">
          <div class="voxcpm-stat">
            <span class="voxcpm-stat-label">Step</span>
            <span class="voxcpm-stat-value" data-stat="step">0/0</span>
          </div>
          <div class="voxcpm-stat">
            <span class="voxcpm-stat-label">Loss</span>
            <span class="voxcpm-stat-value" data-stat="loss">-</span>
          </div>
          <div class="voxcpm-stat">
            <span class="voxcpm-stat-label">LR</span>
            <span class="voxcpm-stat-value" data-stat="lr">-</span>
          </div>
        </div>

        <div class="voxcpm-progress-section">
          <div class="voxcpm-progress-header">
            <span class="voxcpm-progress-label">Training Progress</span>
            <span class="voxcpm-progress-value" data-progress-label>0%</span>
          </div>
          <div class="voxcpm-progress-bar">
            <div class="voxcpm-progress-fill" data-progress-fill></div>
          </div>
        </div>

        <div class="voxcpm-chart-section">
          <div class="voxcpm-chart-header">Loss History</div>
          <canvas class="voxcpm-chart-canvas"></canvas>
        </div>

        <div class="voxcpm-status">
          Ready to train
        </div>

        <div class="voxcpm-preview-section">
          <div class="voxcpm-preview-header">Validation Samples</div>
          <div class="voxcpm-preview-carousel">
            <div class="voxcpm-preview-empty">Audio samples will appear at each checkpoint</div>
          </div>
        </div>
      </div>
    `;

    this.badgeEl = this.element.querySelector(".voxcpm-badge");
    this.stepValueEl = this.element.querySelector('[data-stat="step"]');
    this.lossValueEl = this.element.querySelector('[data-stat="loss"]');
    this.lrValueEl = this.element.querySelector('[data-stat="lr"]');
    this.progressFillEl = this.element.querySelector("[data-progress-fill]");
    this.progressLabelEl = this.element.querySelector("[data-progress-label]");
    this.statusEl = this.element.querySelector(".voxcpm-status");
    this.canvasEl = this.element.querySelector(".voxcpm-chart-canvas");

    const chartSection = this.element.querySelector(".voxcpm-chart-section");
    if (this.canvasEl && chartSection) {
      this.resizeObserver = new ResizeObserver(() => {
        if (this.resizeTimeout) {
          clearTimeout(this.resizeTimeout);
        }
        this.resizeTimeout = window.setTimeout(() => {
          this.drawChart();
        }, 16);
      });
      this.resizeObserver.observe(chartSection);
    }

    this.drawChart();
  }

  updateProgress(step, maxSteps, loss, lr) {
    this.isTraining = true;

    if (this.badgeEl) {
      this.badgeEl.textContent = "Training";
      this.badgeEl.className = "voxcpm-badge training";
    }

    if (this.stepValueEl) {
      this.stepValueEl.textContent = `${step}/${maxSteps}`;
    }
    if (this.lossValueEl) {
      this.lossValueEl.textContent = loss.toFixed(6);
    }
    if (this.lrValueEl) {
      this.lrValueEl.textContent = lr.toExponential(2);
    }

    const progress = (step / maxSteps) * 100;
    if (this.progressFillEl) {
      this.progressFillEl.style.width = `${progress}%`;
    }
    if (this.progressLabelEl) {
      this.progressLabelEl.textContent = `${progress.toFixed(1)}%`;
    }
  }

  updateLossHistory(history) {
    this.lossHistory = history;
    this.drawChart();
  }

  updateStatus(message, type = "normal") {
    if (this.statusEl) {
      this.statusEl.textContent = message;
      this.statusEl.className = "voxcpm-status";
      if (type !== "normal") {
        this.statusEl.classList.add(type);
      }
    }
  }

  onTrainingComplete(finalPath) {
    this.isTraining = false;

    if (this.badgeEl) {
      this.badgeEl.textContent = "Complete";
      this.badgeEl.className = "voxcpm-badge";
      this.badgeEl.style.background = "#22c55e";
    }

    this.updateStatus(`Training complete! Saved to: ${finalPath}`, "success");
  }

  addValidationAudio(audioBase64, checkpointPath) {
    const carousel = this.element.querySelector(".voxcpm-preview-carousel");
    if (!carousel) return;

    // Clear placeholder on first sample
    const empty = carousel.querySelector(".voxcpm-preview-empty");
    if (empty) empty.remove();

    // Extract step label from checkpoint path
    const parts = checkpointPath.replace(/\\/g, "/").split("/");
    const filename = parts[parts.length - 1] || "";
    const stepMatch = filename.match(/step_(\d+)/);
    const shortLabel = stepMatch ? `S${stepMatch[1]}` : filename;

    // Create tile
    const tile = document.createElement("div");
    tile.className = "voxcpm-preview-tile";

    const audio = document.createElement("audio");
    audio.src = audioBase64;
    audio.preload = "auto";

    const playBtn = document.createElement("button");
    playBtn.className = "voxcpm-play-btn";
    playBtn.innerHTML = "&#9654;"; // play triangle

    playBtn.addEventListener("click", () => {
      // Stop any other playing audio in this carousel
      carousel.querySelectorAll("audio").forEach((a) => {
        if (a !== audio) {
          a.pause();
          a.currentTime = 0;
        }
      });
      carousel.querySelectorAll(".voxcpm-play-btn").forEach((b) => {
        if (b !== playBtn) {
          b.classList.remove("playing");
          b.innerHTML = "&#9654;";
        }
      });

      if (audio.paused) {
        audio.play();
        playBtn.classList.add("playing");
        playBtn.innerHTML = "&#9646;&#9646;"; // pause
      } else {
        audio.pause();
        playBtn.classList.remove("playing");
        playBtn.innerHTML = "&#9654;";
      }
    });

    audio.addEventListener("ended", () => {
      playBtn.classList.remove("playing");
      playBtn.innerHTML = "&#9654;";
    });

    const labelEl = document.createElement("div");
    labelEl.className = "tile-label";
    labelEl.textContent = shortLabel;

    tile.appendChild(playBtn);
    tile.appendChild(audio);
    tile.appendChild(labelEl);
    carousel.appendChild(tile);

    // Auto-scroll to newest
    carousel.scrollLeft = carousel.scrollWidth;
  }

  reset() {
    this.lossHistory = [];
    this.isTraining = false;

    if (this.badgeEl) {
      this.badgeEl.textContent = "Training";
      this.badgeEl.className = "voxcpm-badge training";
    }

    if (this.stepValueEl) this.stepValueEl.textContent = "0/0";
    if (this.lossValueEl) this.lossValueEl.textContent = "-";
    if (this.lrValueEl) this.lrValueEl.textContent = "-";

    if (this.progressFillEl) this.progressFillEl.style.width = "0%";
    if (this.progressLabelEl) this.progressLabelEl.textContent = "0%";

    this.updateStatus("Starting training...");
    this.drawChart();

    // Clear audio carousel
    const carousel = this.element.querySelector(".voxcpm-preview-carousel");
    if (carousel) {
      carousel.innerHTML = '<div class="voxcpm-preview-empty">Audio samples will appear at each checkpoint</div>';
    }
  }

  drawChart() {
    if (!this.canvasEl) return;

    const canvas = this.canvasEl;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const w = rect.width;
    const h = rect.height;

    ctx.fillStyle = "#0f0f12";
    ctx.fillRect(0, 0, w, h);

    if (this.lossHistory.length < 2) {
      ctx.fillStyle = "#71717a";
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for training data...", w / 2, h / 2);
      return;
    }

    const padding = { top: 20, right: 20, bottom: 25, left: 50 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    const maxStep = Math.max(...this.lossHistory.map((d) => d.step));
    const minStep = Math.min(...this.lossHistory.map((d) => d.step));
    const maxLoss = Math.max(...this.lossHistory.map((d) => d.loss));
    const minLoss = Math.min(...this.lossHistory.map((d) => d.loss));
    const lossRange = maxLoss - minLoss || 1;

    // Grid
    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 0.5;

    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();

      const lossVal = maxLoss - (lossRange / 4) * i;
      ctx.fillStyle = "#71717a";
      ctx.font = "9px Inter, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(lossVal.toFixed(4), padding.left - 5, y + 3);
    }

    // Loss line
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();

    this.lossHistory.forEach((point, i) => {
      const x = padding.left + ((point.step - minStep) / (maxStep - minStep || 1)) * chartW;
      const y = padding.top + chartH - ((point.loss - minLoss) / lossRange) * chartH;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    gradient.addColorStop(0, "rgba(6, 182, 212, 0.3)");
    gradient.addColorStop(1, "rgba(6, 182, 212, 0)");

    ctx.fillStyle = gradient;
    ctx.beginPath();

    this.lossHistory.forEach((point, i) => {
      const x = padding.left + ((point.step - minStep) / (maxStep - minStep || 1)) * chartW;
      const y = padding.top + chartH - ((point.loss - minLoss) / lossRange) * chartH;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    const lastPoint = this.lossHistory[this.lossHistory.length - 1];
    const lastX = padding.left + ((lastPoint.step - minStep) / (maxStep - minStep || 1)) * chartW;
    ctx.lineTo(lastX, padding.top + chartH);
    ctx.lineTo(padding.left, padding.top + chartH);
    ctx.closePath();
    ctx.fill();

    // Current loss label
    ctx.fillStyle = "#fafafa";
    ctx.font = "bold 10px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`Loss: ${lastPoint.loss.toFixed(6)}`, padding.left + 5, padding.top + 12);
  }

  dispose() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    if (this.resizeTimeout) {
      clearTimeout(this.resizeTimeout);
      this.resizeTimeout = null;
    }
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}

// ---------------------------------------------------------------------------
// ComfyUI Extension Registration
// ---------------------------------------------------------------------------

const widgetInstances = new Map();

function createTrainingWidget(node) {
  const container = document.createElement("div");
  container.id = `voxcpm-training-widget-${node.id}`;
  container.style.width = "100%";
  container.style.height = "100%";
  container.style.minHeight = "280px";

  const widget = node.addDOMWidget(
    "training_ui",
    "training-widget",
    container,
    {
      getMinHeight: () => 400,
      hideOnZoom: false,
      serialize: false,
    }
  );

  setTimeout(() => {
    const trainingWidget = new TrainingWidget({
      node,
      container,
    });
    widgetInstances.set(node.id, trainingWidget);
  }, 100);

  widget.onRemove = () => {
    const instance = widgetInstances.get(node.id);
    if (instance) {
      instance.dispose();
      widgetInstances.delete(node.id);
    }
  };

  return { widget };
}

app.registerExtension({
  name: "ComfyUI.FL_VoxCPM_Training",

  nodeCreated(node) {
    const comfyClass = (node.constructor && node.constructor.comfyClass) || "";

    // Training widget only on LoRA Trainer node
    if (comfyClass !== "FL_VoxCPM_LoRATrainer") {
      return;
    }

    const [oldWidth, oldHeight] = node.size;
    node.setSize([Math.max(oldWidth, 400), Math.max(oldHeight, 500)]);
    createTrainingWidget(node);
  },
});

// Listen for training progress events
api.addEventListener("voxcpm.training.progress", (event) => {
  const detail = event.detail;
  if (!detail || !detail.node) return;

  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (!widget) return;

  switch (detail.type) {
    case "progress":
      widget.updateProgress(
        detail.step ?? 0,
        detail.max_steps ?? 0,
        detail.loss ?? 0,
        detail.lr ?? 0
      );
      if (detail.loss_history) {
        widget.updateLossHistory(detail.loss_history);
      }
      break;

    case "status":
      widget.updateStatus(detail.message ?? "");
      break;

    case "checkpoint":
      if (detail.checkpoint_path) {
        widget.updateStatus(`Checkpoint saved: ${detail.checkpoint_path}`, "success");
      }
      if (detail.validation_audio && detail.checkpoint_path) {
        widget.addValidationAudio(detail.validation_audio, detail.checkpoint_path);
      }
      break;

    case "complete":
      if (detail.final_path) {
        widget.onTrainingComplete(detail.final_path);
      }
      break;
  }
});

// Listen for execution start to reset the widget
api.addEventListener("executing", (event) => {
  const detail = event.detail;
  if (!detail || !detail.node) return;

  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (widget) {
    widget.reset();
  }
});
