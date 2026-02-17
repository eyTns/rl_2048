// ============================================================
// Trainer JS 포팅
// ============================================================

const DEFAULT_CONFIG = {
    gamma: 0.9999,
    learningRate: 0.0001,
    epsilonStart: 0.05,
    epsilonEnd: 0.0001,
    epsilonDecay: 0.99,
};

const INVALID_ACTION_TARGET = -10;

function scaleReward(reward) {
    return reward > 0 ? Math.log2(reward) : 0;
}

// D4 대칭 그룹: [rot_k, flip, actionMap]
const BOARD_AUGMENTATIONS = [
    [0, false, [0, 1, 2, 3]],  // 원본
    [1, false, [2, 3, 1, 0]],  // 90° 반시계
    [2, false, [1, 0, 3, 2]],  // 180°
    [3, false, [3, 2, 0, 1]],  // 270° 반시계
    [0, true,  [0, 1, 3, 2]],  // 좌우 대칭
    [1, true,  [3, 2, 1, 0]],  // 90° + 좌우 대칭
    [2, true,  [1, 0, 2, 3]],  // 180° + 좌우 대칭
    [3, true,  [2, 3, 0, 1]],  // 270° + 좌우 대칭
];

function augmentBoard(board, rotK, flip) {
    let b = board.map(r => [...r]);
    for (let t = 0; t < rotK; t++) {
        const n = b.length;
        const rot = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let x = 0; x < n; x++)
            for (let y = 0; y < n; y++)
                rot[n - 1 - y][x] = b[x][y];  // 반시계 (np.rot90 일치)
        b = rot;
    }
    if (flip) {
        for (let i = 0; i < b.length; i++) b[i].reverse();
    }
    return b;
}

class TDTrainer {
    constructor(model, config = {}) {
        this.model = model;
        this.cfg = { ...DEFAULT_CONFIG, ...config };
        this.epsilon = this.cfg.epsilonStart;
        this.episodeCount = 0;
        this.onStep = null;
        this.aborted = false;
    }

    abort() { this.aborted = true; }

    async trainEpisodes(env, n = 1, onEpisodeEnd = null) {
        for (let ep = 0; ep < n; ep++) {
            if (this.aborted) break;
            const result = await this._trainOne(env);
            this.episodeCount++;
            this.epsilon = Math.max(this.cfg.epsilonEnd, this.epsilon * this.cfg.epsilonDecay);
            if (onEpisodeEnd) onEpisodeEnd({ ...result, episode: this.episodeCount, epsilon: this.epsilon });
        }
    }

    async _trainOne(env) {
        const losses = [];
        let state = env.reset();
        let stepNum = 0;
        let prevStep = null;

        while (!env.done && !this.aborted) {
            const validActions = env.getValidActions();
            const { action, qValues } = this.model.getAction(state, validActions, this.epsilon);
            const { state: nextState, reward, done } = env.step(action);

            // SARSA: 이전 스텝 학습 — target = √r + γ * Q(s', a')
            if (prevStep) {
                const r = scaleReward(prevStep.reward);
                // D4 대칭 8배 증강
                const trainItems = [];
                for (const [rotK, flip, actionMap] of BOARD_AUGMENTATIONS) {
                    const augPrev = augmentBoard(prevStep.state, rotK, flip);
                    const augPrevAction = actionMap[prevStep.action];
                    let target;
                    if (done) {
                        target = r;
                    } else {
                        const augCurr = augmentBoard(state, rotK, flip);
                        const augCurrAction = actionMap[action];
                        const qNext = this.model.forward(augCurr)[augCurrAction];
                        target = r + this.cfg.gamma * qNext;
                    }
                    trainItems.push({ state: augPrev, action: augPrevAction, target });
                }
                let totalLoss = 0;
                for (const item of trainItems) {
                    this.model.forward(item.state);
                    totalLoss += this.model.backward(item.action, item.target, this.cfg.learningRate);
                }
                losses.push(totalLoss / 8);
            }

            // 갈 수 없는 방향: target = 0
            for (let a = 0; a < 4; a++) {
                if (!validActions.includes(a)) {
                    this.model.forward(state);
                    this.model.backward(a, INVALID_ACTION_TARGET, this.cfg.learningRate);
                }
            }

            if (this.onStep) this.onStep({ stepNum, state, action, reward, loss: losses[losses.length - 1] || null, qValues, validActions, done, score: env.score, maxTile: env.getMaxTile() });

            prevStep = { state: state.map(r => [...r]), action, reward };
            state = nextState;
            stepNum++;

            await new Promise(r => setTimeout(r, 0));
        }

        // 게임오버 보드 표시
        if (this.onStep) this.onStep({ stepNum, state, action: -1, reward: 0, loss: null, qValues: this.model.forward(state), done: true, score: env.score, maxTile: env.getMaxTile() });
        await new Promise(r => setTimeout(r, 0));

        // 마지막 스텝: 게임오버 → Q target = 0
        if (prevStep) {
            this.model.forward(prevStep.state);
            losses.push(this.model.backward(prevStep.action, 0, this.cfg.learningRate));
        }

        return { steps: stepNum, score: env.score, maxTile: env.getMaxTile(), losses };
    }
}

class MCTrainer {
    constructor(model, config = {}) {
        this.model = model;
        this.cfg = { ...DEFAULT_CONFIG, ...config };
        this.epsilon = this.cfg.epsilonStart;
        this.episodeCount = 0;
        this.onStep = null;
        this.aborted = false;
    }

    abort() { this.aborted = true; }

    async trainEpisodes(env, n = 1, onEpisodeEnd = null) {
        for (let ep = 0; ep < n; ep++) {
            if (this.aborted) break;
            const result = await this._trainOne(env);
            this.episodeCount++;
            this.epsilon = Math.max(this.cfg.epsilonEnd, this.epsilon * this.cfg.epsilonDecay);
            if (onEpisodeEnd) onEpisodeEnd({ ...result, episode: this.episodeCount, epsilon: this.epsilon });
        }
    }

    async _trainOne(env) {
        const episode = [];
        let state = env.reset();
        let stepNum = 0;

        // 에피소드 수집
        while (!env.done && !this.aborted) {
            const validActions = env.getValidActions();
            const { action, qValues } = this.model.getAction(state, validActions, this.epsilon);
            const { state: nextState, reward, done } = env.step(action);
            episode.push({ state: state.map(r => [...r]), action, reward, done, qValues, validActions });

            if (this.onStep) this.onStep({ stepNum, state, action, reward, loss: null, qValues, validActions, done, score: env.score, maxTile: env.getMaxTile() });

            state = nextState;
            stepNum++;
            if (stepNum % 10 === 0) await new Promise(r => setTimeout(r, 0));
        }

        // 게임오버 보드 표시
        if (this.onStep) this.onStep({ stepNum, state, action: -1, reward: 0, loss: null, qValues: this.model.forward(state), done: true, score: env.score, maxTile: env.getMaxTile() });
        await new Promise(r => setTimeout(r, 0));

        // 중단된 경우 학습하지 않음
        if (this.aborted) return { steps: stepNum, score: env.score, maxTile: env.getMaxTile(), losses: [] };

        // Return 계산 (역순) — √reward + gamma 할인
        // 게임오버 스텝: G = 0 (미래 없음)
        const returns = new Array(episode.length);
        let G = 0;
        for (let i = episode.length - 1; i >= 0; i--) {
            if (episode[i].done) {
                G = 0;
            } else {
                G = scaleReward(episode[i].reward) + this.cfg.gamma * G;
            }
            returns[i] = G;
        }

        // 각 스텝 학습
        const losses = [];
        for (let i = 0; i < episode.length; i++) {
            // D4 대칭 8배 증강
            const trainItems = [];
            for (const [rotK, flip, actionMap] of BOARD_AUGMENTATIONS) {
                const augState = augmentBoard(episode[i].state, rotK, flip);
                const augAction = actionMap[episode[i].action];
                trainItems.push({ state: augState, action: augAction, target: returns[i] });
            }
            let totalLoss = 0;
            for (const item of trainItems) {
                this.model.forward(item.state);
                totalLoss += this.model.backward(item.action, item.target, this.cfg.learningRate);
            }
            losses.push(totalLoss / 8);

            // 갈 수 없는 방향: target = 0
            for (let a = 0; a < 4; a++) {
                if (!episode[i].validActions.includes(a)) {
                    this.model.forward(episode[i].state);
                    this.model.backward(a, INVALID_ACTION_TARGET, this.cfg.learningRate);
                }
            }
        }

        return { steps: stepNum, score: env.score, maxTile: env.getMaxTile(), losses };
    }
}
