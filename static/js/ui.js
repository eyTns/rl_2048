// ============================================================
// UI, 신경망 시각화, 실시간 통합
// ============================================================

let env = new Game2048();
let model = new QNetwork(256);
let currentStep = 0;
let currentEpisode = 0;
let activeTrainer = null;  // 현재 실행 중인 트레이너 (중지용)
let playAborted = false;   // 게임 실행 중지 플래그
let running = false;
let stepMode = false;      // 1스텝 모드 활성 여부
let stepCount = 0;         // 1스텝 모드 스텝 카운터

// --- 게임판 렌더링 (DOM 재사용) ---
function initGrid() {
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    for (let i = 0; i < 16; i++) {
        const tile = document.createElement('div');
        tile.className = 'tile t0';
        grid.appendChild(tile);
    }
}

function renderBoard(board) {
    const tiles = document.getElementById('grid').children;
    const flat = board.flat();
    for (let i = 0; i < 16; i++) {
        const v = flat[i];
        tiles[i].className = 'tile ' + (v > 2048 ? 'tbig' : 't' + v);
        tiles[i].textContent = v || '';
    }
}

// --- 정보 패널 업데이트 ---
function updateInfo(info) {
    if (info.score !== undefined) document.getElementById('infoScore').textContent = info.score;
    if (info.maxTile !== undefined) document.getElementById('infoMaxTile').textContent = info.maxTile;
    if (info.step !== undefined) document.getElementById('infoStep').textContent = info.step;
    if (info.reward !== undefined) document.getElementById('infoReward').textContent = info.reward;
    if (info.epsilon !== undefined) document.getElementById('infoEpsilon').textContent = info.epsilon.toFixed(4);
    if (info.episode !== undefined) document.getElementById('infoEpisode').textContent = info.episode;
}

// --- Q값 바 업데이트 ---
function updateQBars(qValues, bestAction) {
    if (!qValues) return;
    const maxAbs = Math.max(...qValues.map(Math.abs), 0.001);
    for (let a = 0; a < 4; a++) {
        const row = document.getElementById('qRow' + a);
        const fill = row.querySelector('.q-bar-fill');
        const val = row.querySelector('.q-bar-val');
        const pct = Math.abs(qValues[a]) / maxAbs * 100;
        fill.style.width = pct + '%';
        val.textContent = qValues[a].toFixed(4);
        row.className = 'q-bar-row ' + (a === bestAction ? 'q-best' : 'q-normal');
    }
}

// --- 신경망 그래프 ---
function drawNetwork() {
    const canvas = document.getElementById('networkCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
    const H = canvas.height = 220 * (window.devicePixelRatio || 1);
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    const w = canvas.clientWidth, h = 220;
    ctx.clearRect(0, 0, w, h);

    const layers = [16, 16, 16, 4];  // 표시 노드 수
    const layerLabels = ['입력', '은닉1', '은닉2', '출력'];
    const weights = [model.w1, model.w2, model.w3];
    const groupSizes = [1, 16, 16, 1];  // 표시노드당 실제 뉴런 수
    const pad = 40;
    const layerX = layers.map((_, i) => pad + i * (w - 2 * pad) / (layers.length - 1));

    // 노드 y 좌표
    const nodeY = (layerIdx, nodeIdx) => {
        const n = layers[layerIdx];
        const spacing = Math.min(12, (h - 40) / n);
        const totalH = (n - 1) * spacing;
        return (h - 20) / 2 - totalH / 2 + nodeIdx * spacing;
    };

    // 연결선: 레이어별 그룹 평균 가중치 계산 후, 레이어별 최댓값 기준으로 정규화
    const edgeData = [];  // [{ li, si, di, avg }]
    const layerMaxAbs = [];  // 레이어별 |avg| 최댓값

    for (let li = 0; li < weights.length; li++) {
        const W_mat = weights[li];
        const srcN = layers[li], dstN = layers[li + 1];
        const srcGroup = groupSizes[li], dstGroup = groupSizes[li + 1];
        let maxAbs = 0;

        for (let si = 0; si < srcN; si++) {
            for (let di = 0; di < dstN; di++) {
                let sum = 0, count = 0;
                for (let sg = 0; sg < srcGroup; sg++) {
                    for (let dg = 0; dg < dstGroup; dg++) {
                        const srcIdx = si * srcGroup + sg;
                        const dstIdx = di * dstGroup + dg;
                        if (srcIdx < W_mat.length && dstIdx < W_mat[0].length) {
                            sum += W_mat[srcIdx][dstIdx];
                            count++;
                        }
                    }
                }
                if (count === 0) continue;
                const avg = sum / count;
                edgeData.push({ li, si, di, avg });
                const absAvg = Math.abs(avg);
                if (absAvg > maxAbs) maxAbs = absAvg;
            }
        }
        layerMaxAbs.push(maxAbs || 1);
    }

    // 엣지 그리기: 각 레이어의 maxAbs 기준으로 정규화
    for (const { li, si, di, avg } of edgeData) {
        const norm = Math.abs(avg) / layerMaxAbs[li];  // 0~1
        if (norm < 0.02) continue;
        const thickness = Math.min(3, norm * 3);
        const alpha = Math.min(0.8, norm * 0.8);

        ctx.beginPath();
        ctx.moveTo(layerX[li], nodeY(li, si));
        ctx.lineTo(layerX[li + 1], nodeY(li + 1, di));
        ctx.strokeStyle = avg > 0
            ? `rgba(59, 130, 246, ${alpha})`   // 파랑
            : `rgba(239, 68, 68, ${alpha})`;    // 빨강
        ctx.lineWidth = thickness;
        ctx.stroke();
    }

    // 노드
    for (let li = 0; li < layers.length; li++) {
        for (let ni = 0; ni < layers[li]; ni++) {
            const x = layerX[li], y = nodeY(li, ni);
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = '#776e65';
            ctx.fill();
        }
        // 라벨
        ctx.fillStyle = '#999';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(layerLabels[li], layerX[li], h - 4);
    }
}

// --- 스텝 콜백 (학습/게임 중 매 스텝 호출) ---
function onStepUI(info) {
    renderBoard(info.state);
    updateInfo({ score: info.score, maxTile: info.maxTile, step: info.stepNum, reward: info.reward });
    if (info.qValues) {
        const displayQ = [...info.qValues];
        if (info.validActions) {
            for (let a = 0; a < 4; a++) {
                if (!info.validActions.includes(a)) displayQ[a] = 0;
            }
        }
        updateQBars(displayQ, info.action);
    }
}

// --- 에피소드 콜백 ---
function onEpisodeUI(result) {
    updateInfo({ episode: result.episode, epsilon: result.epsilon, score: result.score, maxTile: result.maxTile });
    drawNetwork();
}

// --- 설정 읽기 ---
function readConfig() {
    return {
        gamma: parseFloat(document.getElementById('inputGamma').value) || DEFAULT_CONFIG.gamma,
        learningRate: parseFloat(document.getElementById('inputLearningRate').value) || DEFAULT_CONFIG.learningRate,
        epsilonStart: parseFloat(document.getElementById('inputEpsilonStart').value) || DEFAULT_CONFIG.epsilonStart,
        epsilonEnd: parseFloat(document.getElementById('inputEpsilonEnd').value) || DEFAULT_CONFIG.epsilonEnd,
        epsilonDecay: parseFloat(document.getElementById('inputEpsilonDecay').value) || DEFAULT_CONFIG.epsilonDecay,
        searchDepth: parseInt(document.getElementById('inputSearchDepth').value) || DEFAULT_CONFIG.searchDepth,
    };
}

// --- 버튼 핸들러 ---
function setStatus(text) { document.getElementById('statusText').textContent = text; }
function setRunning(v) {
    running = v;
    document.querySelectorAll('.controls button').forEach(b => b.disabled = v);
    document.getElementById('btnStop').disabled = !v;
}

document.getElementById('btnNewModel').addEventListener('click', () => {
    model = new QNetwork(256);
    setStatus('새 모델 생성');
    initGrid();
    drawNetwork();
    // 에피소드/epsilon 초기화 표시
    updateInfo({ episode: 0, epsilon: readConfig().epsilonStart, score: 0, step: 0, reward: 0, maxTile: 0 });
    updateQBars([0, 0, 0, 0], -1);
});

document.getElementById('btnSave').addEventListener('click', () => {
    const json = JSON.stringify(model.toJSON());
    const blob = new Blob([json], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'model_2048.json';
    a.click();
    URL.revokeObjectURL(a.href);
    setStatus('모델 저장 완료');
});

document.getElementById('btnLoad').addEventListener('click', () => {
    document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        try {
            model = QNetwork.fromJSON(JSON.parse(reader.result));
            setStatus('모델 불러오기 완료');
            drawNetwork();
        } catch (err) {
            setStatus('불러오기 실패: ' + err.message);
        }
    };
    reader.readAsText(file);
    e.target.value = '';
});

document.getElementById('btnPlay').addEventListener('click', async () => {
    if (running) return;
    stepMode = false;
    setRunning(true);
    playAborted = false;
    setStatus('게임 실행 중...');
    env = new Game2048();
    const state = env.reset();
    renderBoard(state);
    let step = 0;
    while (!env.done && !playAborted) {
        const validActions = env.getValidActions();
        if (validActions.length === 0) break;
        const { action, qValues } = model.getAction(env.getState(), validActions, 0);
        const { reward } = env.step(action);
        onStepUI({ stepNum: step, state: env.getState(), action, reward, qValues, validActions, done: env.done, score: env.score, maxTile: env.getMaxTile() });
        step++;
        await new Promise(r => setTimeout(r, 50));
    }
    const msg = playAborted ? '게임 중단' : `게임 종료 · 점수 ${env.score} · 최고 타일 ${env.getMaxTile()}`;
    setStatus(msg);
    setRunning(false);
});

document.getElementById('btnStep').addEventListener('click', () => {
    if (running) return;
    // 첫 클릭이거나 이전 게임이 끝났으면 새 게임 시작
    if (!stepMode || env.done) {
        env = new Game2048();
        env.reset();
        stepCount = 0;
        stepMode = true;
        renderBoard(env.getState());
        updateQBars(model.forward(env.getState()), -1);
        updateInfo({ score: 0, maxTile: env.getMaxTile(), step: 0, reward: 0 });
        setStatus('1스텝 모드 · 클릭하여 진행');
        return;
    }
    // 1스텝 실행
    const validActions = env.getValidActions();
    if (validActions.length === 0) return;
    const { action, qValues } = model.getAction(env.getState(), validActions, 0);
    const { reward } = env.step(action);
    stepCount++;
    onStepUI({ stepNum: stepCount, state: env.getState(), action, reward, qValues, validActions, done: env.done, score: env.score, maxTile: env.getMaxTile() });
    if (env.done) {
        setStatus(`게임 종료 · 점수 ${env.score} · 최고 타일 ${env.getMaxTile()}`);
    } else {
        setStatus(`1스텝 모드 · 스텝 ${stepCount}`);
    }
});

document.getElementById('btnTD').addEventListener('click', async () => {
    if (running) return;
    setRunning(true);
    const n = parseInt(document.getElementById('inputN').value) || 1;
    const curriculum = document.getElementById('chkCurriculum').checked;
    setStatus(`TD 학습 중... (0/${n})${curriculum ? ' [커리큘럼]' : ''}`);
    env = new Game2048(curriculum);
    const trainer = new TDTrainer(model, readConfig());
    trainer.onStep = onStepUI;
    activeTrainer = trainer;
    await trainer.trainEpisodes(env, n, (result) => {
        onEpisodeUI(result);
        setStatus(`TD 학습 중... (${result.episode}/${n})${curriculum ? ' [커리큘럼]' : ''}`);
    });
    const msg = trainer.aborted ? `TD 학습 중단 · ${trainer.episodeCount}게임` : `TD 학습 완료 · ${n}게임`;
    setStatus(msg);
    activeTrainer = null;
    setRunning(false);
});

document.getElementById('btnMC').addEventListener('click', async () => {
    if (running) return;
    setRunning(true);
    const n = parseInt(document.getElementById('inputN').value) || 1;
    const curriculum = document.getElementById('chkCurriculum').checked;
    setStatus(`MC 학습 중... (0/${n})${curriculum ? ' [커리큘럼]' : ''}`);
    env = new Game2048(curriculum);
    const trainer = new MCTrainer(model, readConfig());
    trainer.onStep = onStepUI;
    activeTrainer = trainer;
    await trainer.trainEpisodes(env, n, (result) => {
        onEpisodeUI(result);
        setStatus(`MC 학습 중... (${result.episode}/${n})${curriculum ? ' [커리큘럼]' : ''}`);
    });
    const msg = trainer.aborted ? `MC 학습 중단 · ${trainer.episodeCount}게임` : `MC 학습 완료 · ${n}게임`;
    setStatus(msg);
    activeTrainer = null;
    setRunning(false);
});

document.getElementById('btnTS').addEventListener('click', async () => {
    if (running) return;
    setRunning(true);
    const n = parseInt(document.getElementById('inputN').value) || 1;
    const curriculum = document.getElementById('chkCurriculum').checked;
    const cfg = readConfig();
    setStatus(`TS 학습 중... (0/${n}) [depth=${cfg.searchDepth}]${curriculum ? ' [커리큘럼]' : ''}`);
    env = new Game2048(curriculum);
    const trainer = new TSTrainer(model, cfg);
    trainer.onStep = onStepUI;
    activeTrainer = trainer;
    await trainer.trainEpisodes(env, n, (result) => {
        onEpisodeUI(result);
        setStatus(`TS 학습 중... (${result.episode}/${n}) [depth=${cfg.searchDepth}]${curriculum ? ' [커리큘럼]' : ''}`);
    });
    const msg = trainer.aborted ? `TS 학습 중단 · ${trainer.episodeCount}게임` : `TS 학습 완료 · ${n}게임`;
    setStatus(msg);
    activeTrainer = null;
    setRunning(false);
});

document.getElementById('btnStop').addEventListener('click', () => {
    if (activeTrainer) activeTrainer.abort();
    playAborted = true;
});

// --- 초기화 ---
initGrid();
env.reset();
renderBoard(env.getState());
drawNetwork();
updateInfo({ score: 0, maxTile: 0, step: 0, reward: 0, epsilon: DEFAULT_CONFIG.epsilonStart, episode: 0 });
updateQBars([0, 0, 0, 0], -1);
