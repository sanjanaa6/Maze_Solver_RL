// State Management
const state = {
    algorithm: 'q_learning',
    grid: [],
    rows: 8,
    cols: 8,
    start: [0, 0],
    goal: [7, 7],
    selectedTool: 'wall',
    isDrawing: false,
    policy: null,
    qTable: null,
    evaluatedPath: [],
    trainingResults: null,
    isAnimating: false,
    animationInterval: null,
    isLiveTraining: false,
    liveTrainTimer: null
};

// UI Element References
const canvas = document.getElementById('maze-canvas');
const ctx = canvas.getContext('2d');
const presetSelect = document.getElementById('preset-select');
const btnRandomMaze = document.getElementById('btn-random-maze');
const btnClearWalls = document.getElementById('btn-clear-walls');
const btnTrain = document.getElementById('btn-train');
const btnLiveTrain = document.getElementById('btn-live-train');
const btnAnimate = document.getElementById('btn-animate');
const canvasStatus = document.getElementById('canvas-status');

// Sliders & Labels
const paramAlpha = document.getElementById('param-alpha');
const paramGamma = document.getElementById('param-gamma');
const paramDecay = document.getElementById('param-decay');
const paramEpisodes = document.getElementById('param-episodes');
const paramSpeed = document.getElementById('param-speed');

const valAlpha = document.getElementById('val-alpha');
const valGamma = document.getElementById('val-gamma');
const valDecay = document.getElementById('val-decay');
const valEpisodes = document.getElementById('val-episodes');
const valSpeed = document.getElementById('val-speed');

// Metrics Cards
const metricSteps = document.getElementById('metric-steps');
const metricReward = document.getElementById('metric-reward');
const metricSuccess = document.getElementById('metric-success');
const metricTime = document.getElementById('metric-time');

// Toggles
const chkPolicy = document.getElementById('chk-policy');
const chkHeatmap = document.getElementById('chk-heatmap');

// Chart Instances
let rewardsChart = null;
let stepsChart = null;
let compareChart = null;

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    initChartInstances();
    bindEvents();
    await loadPreset('medium');
});

// Event Binding
function bindEvents() {
    // Segmented Algo Buttons
    document.querySelectorAll('.seg-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.algorithm = e.target.dataset.algo;
            updateTitleDesc();
        });
    });

    // Preset Selection
    presetSelect.addEventListener('change', async (e) => {
        if (e.target.value !== 'custom') {
            await loadPreset(e.target.value);
        }
    });

    // Random Maze Generator
    btnRandomMaze.addEventListener('click', async () => {
        canvasStatus.textContent = 'Generating random maze...';
        try {
            const res = await fetch('/api/maze/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rows: state.rows, cols: state.cols, wall_density: 0.25, trap_density: 0.05 })
            });
            const data = await res.json();
            setMazeState(data);
            presetSelect.value = 'custom';
            canvasStatus.textContent = 'Random Maze Loaded';
        } catch (err) {
            console.error('Maze generation failed:', err);
            canvasStatus.textContent = 'Error generating maze';
        }
    });

    // Clear Canvas
    btnClearWalls.addEventListener('click', () => {
        state.grid = Array(state.rows).fill(0).map(() => Array(state.cols).fill(0));
        state.grid[state.start[0]][state.start[1]] = 2;
        state.grid[state.goal[0]][state.goal[1]] = 3;
        resetTrainingResults();
        drawCanvas();
        presetSelect.value = 'custom';
        canvasStatus.textContent = 'Canvas Cleared';
    });

    // Export Maze JSON
    document.getElementById('btn-export-maze').addEventListener('click', () => {
        const mazeData = {
            rows: state.rows,
            cols: state.cols,
            grid: state.grid,
            start: state.start,
            goal: state.goal
        };
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(mazeData, null, 2));
        const dlAnchor = document.createElement('a');
        dlAnchor.setAttribute("href", dataStr);
        dlAnchor.setAttribute("download", `maze_${state.rows}x${state.cols}.json`);
        document.body.appendChild(dlAnchor);
        dlAnchor.click();
        dlAnchor.remove();
        canvasStatus.textContent = 'Maze Exported to JSON';
    });

    // Import Maze JSON
    const fileImport = document.getElementById('file-import-maze');
    document.getElementById('btn-import-maze').addEventListener('click', () => fileImport.click());

    fileImport.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const data = JSON.parse(event.target.result);
                if (data.grid && data.rows && data.cols && data.start && data.goal) {
                    setMazeState(data);
                    presetSelect.value = 'custom';
                    canvasStatus.textContent = `Imported '${file.name}'`;
                } else {
                    alert('Invalid maze JSON format.');
                }
            } catch (err) {
                alert('Error parsing maze JSON file.');
            }
        };
        reader.readAsText(file);
    });

    // Tool Selector
    document.querySelectorAll('.paint-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.paint-btn').forEach(b => b.classList.remove('active'));
            const target = e.currentTarget;
            target.classList.add('active');
            state.selectedTool = target.dataset.tool;
        });
    });

    // Canvas Interactions (Click & Drag Paint)
    canvas.addEventListener('mousedown', (e) => {
        state.isDrawing = true;
        handleCanvasClick(e);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (state.isDrawing) handleCanvasClick(e);
    });

    window.addEventListener('mouseup', () => {
        state.isDrawing = false;
    });

    // Sliders input sync
    paramAlpha.addEventListener('input', (e) => valAlpha.textContent = parseFloat(e.target.value).toFixed(2));
    paramGamma.addEventListener('input', (e) => valGamma.textContent = parseFloat(e.target.value).toFixed(2));
    paramDecay.addEventListener('input', (e) => valDecay.textContent = parseFloat(e.target.value).toFixed(3));
    paramEpisodes.addEventListener('input', (e) => valEpisodes.textContent = e.target.value);
    paramSpeed.addEventListener('input', (e) => valSpeed.textContent = `${e.target.value} ms`);

    // Toggles
    chkPolicy.addEventListener('change', () => drawCanvas());
    chkHeatmap.addEventListener('change', () => drawCanvas());

    // Training Buttons
    btnTrain.addEventListener('click', runTrainingSession);
    btnLiveTrain.addEventListener('click', toggleLiveTrainingSession);

    // Animate Button
    btnAnimate.addEventListener('click', animatePathExecution);

    // Tab Header Switching
    document.querySelectorAll('.tab-btn').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            const btn = e.currentTarget;
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });
}

function updateTitleDesc() {
    const titleEl = document.getElementById('view-title');
    const descEl = document.getElementById('view-desc');

    if (state.algorithm === 'q_learning') {
        titleEl.textContent = 'Q-Learning Maze Visualizer';
        descEl.textContent = 'Off-Policy temporal difference learning using max Q-value updates.';
    } else if (state.algorithm === 'sarsa') {
        titleEl.textContent = 'SARSA Maze Visualizer';
        descEl.textContent = 'On-Policy temporal difference control evaluating actual next action choice.';
    } else {
        titleEl.textContent = 'Q-Learning vs SARSA Head-to-Head';
        descEl.textContent = 'Benchmarking convergence rate, path optimality, and cumulative reward.';
    }
}

// Load Preset Maze
async function loadPreset(name) {
    canvasStatus.textContent = `Loading preset '${name}'...`;
    try {
        const res = await fetch('/api/presets');
        const data = await res.json();
        if (data[name]) {
            setMazeState(data[name]);
            canvasStatus.textContent = `Preset '${name}' Ready`;
        }
    } catch (err) {
        console.error('Failed to load presets:', err);
    }
}

function setMazeState(data) {
    state.rows = data.rows;
    state.cols = data.cols;
    state.grid = data.grid;
    state.start = data.start;
    state.goal = data.goal;
    resetTrainingResults();
    drawCanvas();
}

function resetTrainingResults() {
    state.policy = null;
    state.qTable = null;
    state.evaluatedPath = [];
    state.trainingResults = null;
    btnAnimate.disabled = true;
    metricSteps.textContent = '--';
    metricReward.textContent = '--';
    metricSuccess.textContent = '--';
    metricTime.textContent = '--';
}

// Canvas Painting Logic
function handleCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellW = canvas.width / state.cols;
    const cellH = canvas.height / state.rows;

    const c = Math.floor(x / cellW);
    const r = Math.floor(y / cellH);

    if (r < 0 || r >= state.rows || c < 0 || c >= state.cols) return;

    if (state.selectedTool === 'wall') {
        if (state.grid[r][c] !== 2 && state.grid[r][c] !== 3) state.grid[r][c] = 1;
    } else if (state.selectedTool === 'empty') {
        if (state.grid[r][c] !== 2 && state.grid[r][c] !== 3) state.grid[r][c] = 0;
    } else if (state.selectedTool === 'trap') {
        if (state.grid[r][c] !== 2 && state.grid[r][c] !== 3) state.grid[r][c] = 4;
    } else if (state.selectedTool === 'start') {
        state.grid[state.start[0]][state.start[1]] = 0;
        state.start = [r, c];
        state.grid[r][c] = 2;
    } else if (state.selectedTool === 'goal') {
        state.grid[state.goal[0]][state.goal[1]] = 0;
        state.goal = [r, c];
        state.grid[r][c] = 3;
    }

    presetSelect.value = 'custom';
    resetTrainingResults();
    drawCanvas();
}

// Canvas Drawing Engine
function drawCanvas() {
    if (!state.grid || state.grid.length === 0) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cellW = w / state.cols;
    const cellH = h / state.rows;

    // Draw Heatmap layer if enabled
    if (chkHeatmap.checked && state.qTable) {
        drawQHeatmap(cellW, cellH);
    }

    // Draw Base Grid Cells
    for (let r = 0; r < state.rows; r++) {
        for (let c = 0; c < state.cols; c++) {
            const val = state.grid[r][c];
            const x = c * cellW;
            const y = r * cellH;

            if (val === 1) { // Wall
                ctx.fillStyle = '#374151';
                ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
            } else if (!chkHeatmap.checked) {
                ctx.fillStyle = '#111827';
                ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
            }

            if (val === 4) { // Trap
                ctx.fillStyle = 'rgba(239, 68, 68, 0.35)';
                ctx.fillRect(x + 2, y + 2, cellW - 4, cellH - 4);
                ctx.fillStyle = '#EF4444';
                ctx.font = `${Math.min(cellW, cellH) * 0.4}px FontAwesome`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('\uf714', x + cellW / 2, y + cellH / 2); // Skull icon
            }

            // Cell Grid Border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.strokeRect(x, y, cellW, cellH);
        }
    }

    // Draw Policy Arrows if enabled
    if (chkPolicy.checked && state.policy) {
        drawPolicyVectors(cellW, cellH);
    }

    // Draw Evaluated Path Line
    if (state.evaluatedPath.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = Math.max(3, cellW * 0.1);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        state.evaluatedPath.forEach((pt, i) => {
            const px = pt[1] * cellW + cellW / 2;
            const py = pt[0] * cellH + cellH / 2;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });
        ctx.stroke();
    }

    // Draw Start (S) Cell
    const sx = state.start[1] * cellW;
    const sy = state.start[0] * cellH;
    ctx.fillStyle = '#38BDF8';
    ctx.fillRect(sx + 3, sy + 3, cellW - 6, cellH - 6);
    ctx.fillStyle = '#000';
    ctx.font = `bold ${Math.min(cellW, cellH) * 0.45}px Outfit`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('S', sx + cellW / 2, sy + cellH / 2);

    // Draw Goal (G) Cell
    const gx = state.goal[1] * cellW;
    const gy = state.goal[0] * cellH;
    ctx.fillStyle = '#F59E0B';
    ctx.fillRect(gx + 3, gy + 3, cellW - 6, cellH - 6);
    ctx.fillStyle = '#000';
    ctx.font = `bold ${Math.min(cellW, cellH) * 0.45}px Outfit`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('G', gx + cellW / 2, gy + cellH / 2);
}

// Q-Table Max Value Heatmap Overlay
function drawQHeatmap(cellW, cellH) {
    let maxQ = -Infinity;
    let minQ = Infinity;

    // Find min and max Q values
    Object.values(state.qTable).forEach(qArr => {
        const m = Math.max(...qArr);
        if (m > maxQ) maxQ = m;
        if (m < minQ) minQ = m;
    });

    const range = (maxQ - minQ) || 1;

    for (let r = 0; r < state.rows; r++) {
        for (let c = 0; c < state.cols; c++) {
            if (state.grid[r][c] === 1) continue;
            const key = `${r},${c}`;
            const qVals = state.qTable[key] || [0, 0, 0, 0];
            const maxVal = Math.max(...qVals);
            
            // Normalize between 0 and 1
            const norm = (maxVal - minQ) / range;
            const hue = 240 - (norm * 180); // Blue to Green/Yellow
            
            ctx.fillStyle = `hsla(${hue}, 80%, 45%, 0.4)`;
            ctx.fillRect(c * cellW + 1, r * cellH + 1, cellW - 2, cellH - 2);
        }
    }
}

// Policy Vector Arrows Overlay
function drawPolicyVectors(cellW, cellH) {
    const arrowSymbols = ['\u2191', '\u2192', '\u2193', '\u2190']; // Up, Right, Down, Left

    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = `${Math.min(cellW, cellH) * 0.35}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let r = 0; r < state.rows; r++) {
        for (let c = 0; c < state.cols; c++) {
            if (state.grid[r][c] === 1 || (r === state.goal[0] && c === state.goal[1])) continue;
            const key = `${r},${c}`;
            const action = state.policy ? state.policy[key] : null;
            if (action !== null && action !== undefined) {
                const x = c * cellW + cellW / 2;
                const y = r * cellH + cellH / 2;
                ctx.fillText(arrowSymbols[action], x, y);
            }
        }
    }
}

// Run Training Request
async function runTrainingSession() {
    const algo = state.algorithm;
    const episodes = parseInt(paramEpisodes.value);
    const alpha = parseFloat(paramAlpha.value);
    const gamma = parseFloat(paramGamma.value);
    const epsilon_decay = parseFloat(paramDecay.value);

    btnTrain.disabled = true;
    canvasStatus.textContent = 'Training in progress...';

    const mazeConfig = {
        grid: state.grid,
        rows: state.rows,
        cols: state.cols,
        start_pos: state.start,
        goal_pos: state.goal
    };

    try {
        if (algo === 'compare') {
            const res = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ maze: mazeConfig, episodes, alpha, gamma, epsilon_decay })
            });
            const data = await res.json();
            renderComparisonResults(data);
        } else {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ algorithm: algo, maze: mazeConfig, episodes, alpha, gamma, epsilon_decay })
            });
            const data = await res.json();
            renderSingleTrainResults(data);
        }
        canvasStatus.textContent = 'Training Complete';
    } catch (err) {
        console.error('Training failed:', err);
        canvasStatus.textContent = 'Training Error';
    } finally {
        btnTrain.disabled = false;
    }
}

// Render Single Algorithm Results
function renderSingleTrainResults(data) {
    state.trainingResults = data;
    state.policy = data.policy;
    state.qTable = data.q_table;
    state.evaluatedPath = data.eval.path;

    btnAnimate.disabled = false;

    // Metrics Update
    metricSteps.textContent = data.eval.success ? data.eval.steps : 'Failed';
    metricReward.textContent = data.eval.total_reward.toFixed(1);
    
    const last50 = data.success_history.slice(-50);
    const succRate = (last50.reduce((a, b) => a + b, 0) / last50.length * 100).toFixed(1);
    metricSuccess.textContent = `${succRate}%`;
    metricTime.textContent = `${data.training_time_ms} ms`;

    // Charts Update
    updateChartsSingle(data);
    drawCanvas();
}

// Render Comparison Results
function renderComparisonResults(data) {
    const qData = data.q_learning;
    const sarsaData = data.sarsa;
    const summary = data.summary;

    // Display Q-Learning path as primary canvas overlay
    state.policy = qData.policy;
    state.qTable = qData.q_table;
    state.evaluatedPath = qData.eval.path;
    btnAnimate.disabled = false;

    // Metrics Update
    metricSteps.textContent = `Q: ${summary.q_learning.eval_path_length - 1} | S: ${summary.sarsa.eval_path_length - 1}`;
    metricReward.textContent = `Q: ${summary.q_learning.avg_reward_last50} | S: ${summary.sarsa.avg_reward_last50}`;
    metricSuccess.textContent = `Q: ${summary.q_learning.success_rate}% | S: ${summary.sarsa.success_rate}%`;
    metricTime.textContent = `Q: ${summary.q_learning.time_ms}ms | S: ${summary.sarsa.time_ms}ms`;

    // Populate Comparison Table
    const tbody = document.getElementById('compare-table-body');
    tbody.innerHTML = `
        <tr><td>Avg Steps (Last 50)</td><td>${summary.q_learning.avg_steps_last50}</td><td>${summary.sarsa.avg_steps_last50}</td></tr>
        <tr><td>Avg Reward (Last 50)</td><td>${summary.q_learning.avg_reward_last50}</td><td>${summary.sarsa.avg_reward_last50}</td></tr>
        <tr><td>Success Rate (Last 50)</td><td>${summary.q_learning.success_rate}%</td><td>${summary.sarsa.success_rate}%</td></tr>
        <tr><td>Evaluated Path Length</td><td>${summary.q_learning.eval_path_length - 1} steps</td><td>${summary.sarsa.eval_path_length - 1} steps</td></tr>
        <tr><td>Training Execution Time</td><td>${summary.q_learning.time_ms} ms</td><td>${summary.sarsa.time_ms} ms</td></tr>
    `;

    // Switch to Comparison Tab
    document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector('[data-tab="compare-tab"]').classList.add('active');
    document.getElementById('compare-tab').classList.add('active');

    updateChartsComparison(qData, sarsaData);
    drawCanvas();
}

// Chart.js Setup
function initChartInstances() {
    const ctxRewards = document.getElementById('rewards-chart').getContext('2d');
    const ctxSteps = document.getElementById('steps-chart').getContext('2d');
    const ctxCompare = document.getElementById('compare-chart').getContext('2d');

    const commonOpts = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#9CA3AF', font: { family: 'Outfit' } } } },
        scales: {
            x: { ticks: { color: '#6B7280' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { ticks: { color: '#6B7280' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
    };

    rewardsChart = new Chart(ctxRewards, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...commonOpts, plugins: { ...commonOpts.plugins, title: { display: true, text: 'Cumulative Episode Reward', color: '#F9FAFB' } } }
    });

    stepsChart = new Chart(ctxSteps, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...commonOpts, plugins: { ...commonOpts.plugins, title: { display: true, text: 'Steps Taken per Episode', color: '#F9FAFB' } } }
    });

    compareChart = new Chart(ctxCompare, {
        type: 'bar',
        data: { labels: ['Success Rate (%)', 'Path Steps (Shortest)'], datasets: [] },
        options: { ...commonOpts, plugins: { ...commonOpts.plugins, title: { display: true, text: 'Head-to-Head Comparison Summary', color: '#F9FAFB' } } }
    });
}

function updateChartsSingle(data) {
    const labels = data.episode_rewards.map((_, i) => i + 1);

    rewardsChart.data.labels = labels;
    rewardsChart.data.datasets = [{
        label: 'Episode Reward',
        data: data.episode_rewards,
        borderColor: '#38BDF8',
        backgroundColor: 'rgba(56, 189, 248, 0.1)',
        tension: 0.2,
        pointRadius: 0
    }];
    rewardsChart.update();

    stepsChart.data.labels = labels;
    stepsChart.data.datasets = [{
        label: 'Steps Taken',
        data: data.episode_steps,
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.2,
        pointRadius: 0
    }];
    stepsChart.update();
}

function updateChartsComparison(qData, sarsaData) {
    const labels = qData.episode_rewards.map((_, i) => i + 1);

    rewardsChart.data.labels = labels;
    rewardsChart.data.datasets = [
        { label: 'Q-Learning Reward', data: qData.episode_rewards, borderColor: '#38BDF8', pointRadius: 0 },
        { label: 'SARSA Reward', data: sarsaData.episode_rewards, borderColor: '#10B981', pointRadius: 0 }
    ];
    rewardsChart.update();

    stepsChart.data.labels = labels;
    stepsChart.data.datasets = [
        { label: 'Q-Learning Steps', data: qData.episode_steps, borderColor: '#38BDF8', pointRadius: 0 },
        { label: 'SARSA Steps', data: sarsaData.episode_steps, borderColor: '#10B981', pointRadius: 0 }
    ];
    stepsChart.update();

    const qSucc = (qData.success_history.slice(-50).reduce((a,b)=>a+b,0)/50)*100;
    const sarsaSucc = (sarsaData.success_history.slice(-50).reduce((a,b)=>a+b,0)/50)*100;

    compareChart.data.datasets = [
        { label: 'Q-Learning', data: [qSucc, qData.eval.path.length - 1], backgroundColor: 'rgba(56, 189, 248, 0.7)' },
        { label: 'SARSA', data: [sarsaSucc, sarsaData.eval.path.length - 1], backgroundColor: 'rgba(16, 185, 129, 0.7)' }
    ];
    compareChart.update();
}

// Animate Agent Path Traversal
function animatePathExecution() {
    if (!state.evaluatedPath || state.evaluatedPath.length === 0) return;

    if (state.isAnimating) {
        clearInterval(state.animationInterval);
        state.isAnimating = false;
        btnAnimate.innerHTML = '<i class="fa-solid fa-person-running"></i> Animate Path';
        drawCanvas();
        return;
    }

    state.isAnimating = true;
    btnAnimate.innerHTML = '<i class="fa-solid fa-pause"></i> Pause Animation';

    let stepIdx = 0;
    const path = state.evaluatedPath;
    const cellW = canvas.width / state.cols;
    const cellH = canvas.height / state.rows;

    state.animationInterval = setInterval(() => {
        if (stepIdx >= path.length) {
            clearInterval(state.animationInterval);
            state.isAnimating = false;
            btnAnimate.innerHTML = '<i class="fa-solid fa-person-running"></i> Animate Path';
            drawCanvas();
            return;
        }

        drawCanvas();

        // Draw glowing agent circle at current step
        const curr = path[stepIdx];
        const cx = curr[1] * cellW + cellW / 2;
        const cy = curr[0] * cellH + cellH / 2;

        ctx.shadowColor = '#10B981';
        ctx.shadowBlur = 15;
        ctx.fillStyle = '#10B981';
        ctx.beginPath();
        ctx.arc(cx, cy, Math.min(cellW, cellH) * 0.3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;

        stepIdx++;
    }, 200);
}

// Live Interactive Step-by-Step Training Session
async function toggleLiveTrainingSession() {
    if (state.isLiveTraining) {
        state.isLiveTraining = false;
        clearTimeout(state.liveTrainTimer);
        btnLiveTrain.innerHTML = '<i class="fa-solid fa-bolt"></i> Live Visual Train';
        canvasStatus.textContent = 'Live Training Paused';
        return;
    }

    const algo = state.algorithm === 'compare' ? 'q_learning' : state.algorithm;
    const episodes = parseInt(paramEpisodes.value);
    const alpha = parseFloat(paramAlpha.value);
    const gamma = parseFloat(paramGamma.value);
    let epsilon = 1.0;
    const epsilon_decay = parseFloat(paramDecay.value);

    state.isLiveTraining = true;
    btnLiveTrain.innerHTML = '<i class="fa-solid fa-square"></i> Stop Live Train';

    // Initialize state Q-table & policy
    state.qTable = {};
    state.policy = {};
    for (let r = 0; r < state.rows; r++) {
        for (let c = 0; c < state.cols; c++) {
            state.qTable[`${r},${c}`] = [0, 0, 0, 0];
            state.policy[`${r},${c}`] = 0;
        }
    }

    const episodeRewards = [];
    const episodeSteps = [];
    const successHistory = [];
    const actionsVec = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // Up, Right, Down, Left

    let currentEp = 0;
    const startTime = Date.now();

    function chooseAction(r, c, eps) {
        if (Math.random() < eps) return Math.floor(Math.random() * 4);
        const qVals = state.qTable[`${r},${c}`];
        const maxQ = Math.max(...qVals);
        const bests = [];
        for (let a = 0; a < 4; a++) if (qVals[a] === maxQ) bests.push(a);
        return bests[Math.floor(Math.random() * bests.length)];
    }

    function runEpisodeStep() {
        if (!state.isLiveTraining || currentEp >= episodes) {
            state.isLiveTraining = false;
            btnLiveTrain.innerHTML = '<i class="fa-solid fa-bolt"></i> Live Visual Train';
            canvasStatus.textContent = 'Live Training Complete';
            
            extractGreedyPath();
            btnAnimate.disabled = false;
            return;
        }

        let currR = state.start[0];
        let currC = state.start[1];
        let epReward = 0;
        let epSteps = 0;
        let done = false;
        let reachedGoal = false;
        const maxSteps = state.rows * state.cols * 4;

        let currA = chooseAction(currR, currC, epsilon);

        while (!done && epSteps < maxSteps) {
            if (algo === 'q_learning') {
                currA = chooseAction(currR, currC, epsilon);
            }

            const move = actionsVec[currA];
            let nextR = currR + move[0];
            let nextC = currC + move[1];

            let reward = -1.0;

            if (nextR < 0 || nextR >= state.rows || nextC < 0 || nextC >= state.cols || state.grid[nextR][nextC] === 1) {
                nextR = currR;
                nextC = currC;
                reward = -5.0;
            } else if (state.grid[nextR][nextC] === 3) {
                reward = 100.0;
                done = true;
                reachedGoal = true;
            } else if (state.grid[nextR][nextC] === 4) {
                reward = -20.0;
            }

            const currKey = `${currR},${currC}`;
            const nextKey = `${nextR},${nextC}`;

            let nextA = null;
            if (algo === 'sarsa') {
                nextA = !done ? chooseAction(nextR, nextC, epsilon) : null;
            }

            const target = done ? reward : reward + gamma * (algo === 'q_learning' ? Math.max(...state.qTable[nextKey]) : state.qTable[nextKey][nextA]);
            state.qTable[currKey][currA] += alpha * (target - state.qTable[currKey][currA]);

            state.policy[currKey] = state.qTable[currKey].indexOf(Math.max(...state.qTable[currKey]));

            epReward += reward;
            epSteps++;

            currR = nextR;
            currC = nextC;
            if (algo === 'sarsa') currA = nextA;
        }

        epsilon = Math.max(0.01, epsilon * epsilon_decay);
        currentEp++;

        episodeRewards.push(epReward);
        episodeSteps.push(epSteps);
        successHistory.push(reachedGoal ? 1 : 0);

        if (currentEp % Math.max(1, Math.floor(episodes / 50)) === 0 || currentEp === episodes) {
            canvasStatus.textContent = `Live Ep ${currentEp}/${episodes} | Epsilon: ${epsilon.toFixed(2)}`;
            metricSteps.textContent = epSteps;
            metricReward.textContent = epReward.toFixed(1);
            const succ50 = successHistory.slice(-50);
            metricSuccess.textContent = `${((succ50.reduce((a,b)=>a+b,0)/succ50.length)*100).toFixed(1)}%`;
            metricTime.textContent = `${Date.now() - startTime} ms`;

            updateChartsSingle({ episode_rewards: episodeRewards, episode_steps: episodeSteps });
            drawCanvas();
        }

        const delay = parseInt(paramSpeed.value);
        state.liveTrainTimer = setTimeout(runEpisodeStep, delay);
    }

    runEpisodeStep();
}

function extractGreedyPath() {
    let r = state.start[0];
    let c = state.start[1];
    const path = [[r, c]];
    const maxSteps = state.rows * state.cols * 2;
    let steps = 0;

    while (steps < maxSteps) {
        if (r === state.goal[0] && c === state.goal[1]) break;
        const key = `${r},${c}`;
        const action = state.policy ? state.policy[key] : null;
        if (action === null || action === undefined) break;

        const move = [[-1, 0], [0, 1], [1, 0], [0, -1]][action];
        const nr = r + move[0];
        const nc = c + move[1];

        if (nr < 0 || nr >= state.rows || nc < 0 || nc >= state.cols || state.grid[nr][nc] === 1) break;

        r = nr;
        c = nc;
        path.push([r, c]);
        steps++;
    }

    state.evaluatedPath = path;
    drawCanvas();
}
