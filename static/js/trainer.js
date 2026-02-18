// ============================================================
// Trainer JS 포팅
// ============================================================

const DEFAULT_CONFIG = {
    gamma: 0.9999,
    learningRate: 0.0001,
    epsilonStart: 0.05,
    epsilonEnd: 0.0001,
    epsilonDecay: 0.99,
    searchDepth: 2,  // TS용: tree search 깊이
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

    // 1-step TD + D4 증강 학습
    _learnStep(state, action, reward, nextState, done) {
        const trainItems = [];
        for (const [rotK, flip, actionMap] of BOARD_AUGMENTATIONS) {
            const augState = augmentBoard(state, rotK, flip);
            const augAction = actionMap[action];
            let target;
            if (done) {
                // 게임오버: terminal value = -10
                target = scaleReward(reward) + this.cfg.gamma * INVALID_ACTION_TARGET;
            } else {
                const augNext = augmentBoard(nextState, rotK, flip);
                const qNext = this.model.forward(augNext);
                // augmented nextState에서 valid action만 max
                const sim = new Game2048();
                sim.board = augNext.map(r => [...r]);
                const augValidActions = sim.getValidActions();
                let maxQ = INVALID_ACTION_TARGET;
                for (const va of augValidActions) {
                    if (qNext[va] > maxQ) maxQ = qNext[va];
                }
                target = scaleReward(reward) + this.cfg.gamma * maxQ;
            }
            trainItems.push({ state: augState, action: augAction, target });
        }
        let totalLoss = 0;
        for (const item of trainItems) {
            this.model.forward(item.state);
            totalLoss += this.model.backward(item.action, item.target, this.cfg.learningRate);
        }
        return totalLoss / 8;
    }

    async _trainOne(env) {
        const losses = [];
        let state = env.reset();
        let stepNum = 0;

        while (!env.done && !this.aborted) {
            const validActions = env.getValidActions();
            const { action, qValues } = this.model.getAction(state, validActions, this.epsilon);
            const { state: nextState, reward, done } = env.step(action);

            // 1-step TD 학습
            const loss = this._learnStep(state.map(r => [...r]), action, reward, nextState, done);
            losses.push(loss);

            // 갈 수 없는 방향: 증강 8개 state 전부에 -10 학습
            for (const [rotK, flip, actionMap] of BOARD_AUGMENTATIONS) {
                const augState = augmentBoard(state, rotK, flip);
                for (let a = 0; a < 4; a++) {
                    if (!validActions.includes(a)) {
                        const augInvalidAction = actionMap[a];
                        this.model.forward(augState);
                        this.model.backward(augInvalidAction, INVALID_ACTION_TARGET, this.cfg.learningRate);
                    }
                }
            }

            if (this.onStep) this.onStep({ stepNum, state, action, reward, loss, qValues, validActions, done, score: env.score, maxTile: env.getMaxTile() });

            state = nextState;
            stepNum++;

            await new Promise(r => setTimeout(r, 0));
        }

        // 게임오버 보드 표시
        if (this.onStep) this.onStep({ stepNum, state, action: -1, reward: 0, loss: null, qValues: this.model.forward(state), done: true, score: env.score, maxTile: env.getMaxTile() });
        await new Promise(r => setTimeout(r, 0));

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
        // 게임오버 스텝: terminal value = -10
        const returns = new Array(episode.length);
        let G = INVALID_ACTION_TARGET;
        for (let i = episode.length - 1; i >= 0; i--) {
            if (episode[i].done) {
                G = scaleReward(episode[i].reward) + this.cfg.gamma * INVALID_ACTION_TARGET;
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

            // 갈 수 없는 방향: 증강 8개 state 전부에 -10 학습
            for (const [rotK, flip, actionMap] of BOARD_AUGMENTATIONS) {
                const augState = augmentBoard(episode[i].state, rotK, flip);
                for (let a = 0; a < 4; a++) {
                    if (!episode[i].validActions.includes(a)) {
                        const augInvalidAction = actionMap[a];
                        this.model.forward(augState);
                        this.model.backward(augInvalidAction, INVALID_ACTION_TARGET, this.cfg.learningRate);
                    }
                }
            }
        }

        return { steps: stepNum, score: env.score, maxTile: env.getMaxTile(), losses };
    }
}


class TSTrainer {
    constructor(model, config = {}) {
        this.model = model;
        this.cfg = { ...DEFAULT_CONFIG, ...config };
        this.epsilon = this.cfg.epsilonStart;
        this.episodeCount = 0;
        this.onStep = null;
        this.aborted = false;
        this.searchDepth = this.cfg.searchDepth;
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

    // k-step tree search: 모든 유효 행동을 재귀적으로 시뮬레이션
    _treeSearch(board, depth) {
        const sim = new Game2048();
        sim.board = board.map(r => [...r]);
        sim.done = !sim._canMove();

        const validActions = sim.getValidActions();
        if (validActions.length === 0) return { value: INVALID_ACTION_TARGET, action: -1 };

        if (depth === 0) {
            // 리프 노드: Q-network로 평가
            const qValues = this.model.forward(board);
            let bestA = validActions[0], bestQ = -Infinity;
            for (const a of validActions) {
                if (qValues[a] > bestQ) { bestQ = qValues[a]; bestA = a; }
            }
            return { value: bestQ, action: bestA };
        }

        let bestValue = -Infinity;
        let bestAction = validActions[0];

        for (const action of validActions) {
            // 환경 복제 후 시뮬레이션 (타일 스폰 포함)
            const simEnv = new Game2048();
            simEnv.board = board.map(r => [...r]);
            simEnv.done = false;
            const { reward, done } = simEnv.step(action);
            const r = scaleReward(reward);

            let value;
            if (done) {
                value = r + this.cfg.gamma * INVALID_ACTION_TARGET;
            } else {
                const child = this._treeSearch(simEnv.board, depth - 1);
                value = r + this.cfg.gamma * child.value;
            }

            if (value > bestValue) {
                bestValue = value;
                bestAction = action;
            }
        }

        return { value: bestValue, action: bestAction };
    }

    // Tree search로 행동 선택 (epsilon 탐험 유지)
    _selectAction(state, validActions) {
        if (Math.random() < this.epsilon) {
            const action = validActions[Math.floor(Math.random() * validActions.length)];
            return { action, qValues: this.model.forward(state) };
        }
        const result = this._treeSearch(state, this.searchDepth);
        return { action: result.action, qValues: this.model.forward(state) };
    }

    // 1-step TD + D4 증강 학습
    _learnStep(state, action, reward, nextState, done) {
        const trainItems = [];
        for (const [rotK, flip, actionMap] of BOARD_AUGMENTATIONS) {
            const augState = augmentBoard(state, rotK, flip);
            const augAction = actionMap[action];
            let target;
            if (done) {
                // 게임오버: terminal value = -10
                target = scaleReward(reward) + this.cfg.gamma * INVALID_ACTION_TARGET;
            } else {
                const augNext = augmentBoard(nextState, rotK, flip);
                const qNext = this.model.forward(augNext);
                // augmented nextState에서 valid action만 max
                const sim = new Game2048();
                sim.board = augNext.map(r => [...r]);
                const augValidActions = sim.getValidActions();
                let maxQ = INVALID_ACTION_TARGET;
                for (const va of augValidActions) {
                    if (qNext[va] > maxQ) maxQ = qNext[va];
                }
                target = scaleReward(reward) + this.cfg.gamma * maxQ;
            }
            trainItems.push({ state: augState, action: augAction, target });
        }
        let totalLoss = 0;
        for (const item of trainItems) {
            this.model.forward(item.state);
            totalLoss += this.model.backward(item.action, item.target, this.cfg.learningRate);
        }
        return totalLoss / 8;
    }

    async _trainOne(env) {
        const losses = [];
        let state = env.reset();
        let stepNum = 0;

        while (!env.done && !this.aborted) {
            const validActions = env.getValidActions();
            const { action, qValues } = this._selectAction(state, validActions);
            const { state: nextState, reward, done } = env.step(action);

            // 1-step TD 학습
            const loss = this._learnStep(state.map(r => [...r]), action, reward, nextState, done);
            losses.push(loss);

            // 갈 수 없는 방향: 증강 8개 state 전부에 -10 학습
            for (const [rotK, flip, actionMap] of BOARD_AUGMENTATIONS) {
                const augState = augmentBoard(state, rotK, flip);
                for (let a = 0; a < 4; a++) {
                    if (!validActions.includes(a)) {
                        const augInvalidAction = actionMap[a];
                        this.model.forward(augState);
                        this.model.backward(augInvalidAction, INVALID_ACTION_TARGET, this.cfg.learningRate);
                    }
                }
            }

            if (this.onStep) this.onStep({ stepNum, state, action, reward, loss, qValues, validActions, done, score: env.score, maxTile: env.getMaxTile() });

            state = nextState;
            stepNum++;

            await new Promise(r => setTimeout(r, 0));
        }

        // 게임오버 보드 표시
        if (this.onStep) this.onStep({ stepNum, state, action: -1, reward: 0, loss: null, qValues: this.model.forward(state), done: true, score: env.score, maxTile: env.getMaxTile() });
        await new Promise(r => setTimeout(r, 0));

        return { steps: stepNum, score: env.score, maxTile: env.getMaxTile(), losses };
    }
}
