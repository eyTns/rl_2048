// ============================================================
// QNetwork JS 포팅
// ============================================================

const NUM_CHANNELS = 16;  // 원핫 채널 수: exponent 1~16 (2^1 ~ 2^16)
const INPUT_SIZE = 4 * 4 * NUM_CHANNELS;  // 256 (16셀 × 16채널)
const GRAD_NORM_LIMIT = 1.0;
const HUBER_DELTA = 1.0;

// --- 행렬 유틸 ---
function zeros(rows, cols) {
    if (cols === undefined) return new Float64Array(rows);
    return Array.from({ length: rows }, () => new Float64Array(cols));
}

function matMul(A, B) {
    // A: [m][k], B: [k][n] → C: [m][n]
    const m = A.length, k = B.length, n = B[0].length;
    const C = Array.from({ length: m }, () => new Float64Array(n));
    for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) {
            let s = 0;
            for (let p = 0; p < k; p++) s += A[i][p] * B[p][j];
            C[i][j] = s;
        }
    return C;
}

function transpose(A) {
    const m = A.length, n = A[0].length;
    const T = Array.from({ length: n }, () => new Float64Array(m));
    for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++)
            T[j][i] = A[i][j];
    return T;
}

function addBias(A, b) {
    // A: [m][n], b: Float64Array(n) → A[i][j] + b[j]
    const m = A.length, n = A[0].length;
    const R = Array.from({ length: m }, () => new Float64Array(n));
    for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++)
            R[i][j] = A[i][j] + b[j];
    return R;
}

function randn() {
    // Box-Muller
    const u1 = Math.random(), u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

class QNetwork {
    constructor(hiddenSize = 256) {
        this.hiddenSize = hiddenSize;
        const h = hiddenSize;
        const xavier = (fanIn) => Math.sqrt(2.0 / fanIn);

        this.w1 = this._randMatrix(INPUT_SIZE, h, xavier(INPUT_SIZE));
        this.b1 = zeros(h);
        this.w2 = this._randMatrix(h, h, xavier(h));
        this.b2 = zeros(h);
        this.w3 = this._randMatrix(h, 4, xavier(h));
        this.b3 = zeros(4);

        this._cache = {};
    }

    _randMatrix(rows, cols, scale) {
        const M = Array.from({ length: rows }, () => new Float64Array(cols));
        for (let i = 0; i < rows; i++)
            for (let j = 0; j < cols; j++)
                M[i][j] = randn() * scale;
        return M;
    }

    _preprocess(board) {
        // board: 4x4 array → [1][256] 원핫 인코딩
        const x = [new Float64Array(INPUT_SIZE)];
        for (let i = 0; i < 4; i++)
            for (let j = 0; j < 4; j++) {
                const v = board[i][j];
                if (v > 0) {
                    const exp = Math.log2(v);  // 2→1, 4→2, ..., 65536→16
                    const cell = i * 4 + j;
                    if (exp >= 1 && exp <= NUM_CHANNELS) {
                        x[0][cell * NUM_CHANNELS + exp - 1] = 1.0;
                    }
                }
            }
        return x;
    }

    forward(board) {
        const x = this._preprocess(board);
        const z1 = addBias(matMul(x, this.w1), this.b1);
        const a1 = z1.map(r => r.map(v => Math.max(0, v)));  // ReLU
        const z2 = addBias(matMul(a1, this.w2), this.b2);
        const a2 = z2.map(r => r.map(v => Math.max(0, v)));
        const z3 = addBias(matMul(a2, this.w3), this.b3);

        this._cache = { x, z1, a1, z2, a2, z3 };
        return Array.from(z3[0]);  // [4] Q-values
    }

    backward(action, target, lr = 0.001) {
        const { x, z1, a1, z2, a2, z3 } = this._cache;
        const qVal = z3[0][action];
        if (isNaN(qVal)) return 0;

        const error = qVal - target;
        const absError = Math.abs(error);
        let loss, dloss;
        if (absError <= HUBER_DELTA) {
            loss = 0.5 * error * error;
            dloss = error;
        } else {
            loss = HUBER_DELTA * (absError - 0.5 * HUBER_DELTA);
            dloss = HUBER_DELTA * Math.sign(error);
        }

        // dz3: [1][4]
        const dz3 = [new Float64Array(4)];
        dz3[0][action] = dloss;

        // Layer 3
        const dw3 = matMul(transpose(a2), dz3);
        const db3 = new Float64Array(dz3[0]);
        const da2 = matMul(dz3, transpose(this.w3));

        // Layer 2 (ReLU backward)
        const dz2 = da2.map((r, i) => r.map((v, j) => z2[i][j] > 0 ? v : 0));
        const dw2 = matMul(transpose(a1), dz2);
        const db2 = new Float64Array(dz2[0]);
        const da1 = matMul(dz2, transpose(this.w2));

        // Layer 1
        const dz1 = da1.map((r, i) => r.map((v, j) => z1[i][j] > 0 ? v : 0));
        const dw1 = matMul(transpose(x), dz1);
        const db1 = new Float64Array(dz1[0]);

        // Clip & update
        this._updateWeights(this.w3, dw3, this.b3, db3, lr);
        this._updateWeights(this.w2, dw2, this.b2, db2, lr);
        this._updateWeights(this.w1, dw1, this.b1, db1, lr);

        return loss;
    }

    _clipNorm(g) {
        // g: 2D array or 1D Float64Array → norm clip
        let sumSq = 0;
        if (g[0] instanceof Float64Array || Array.isArray(g[0])) {
            for (let i = 0; i < g.length; i++)
                for (let j = 0; j < g[0].length; j++)
                    sumSq += g[i][j] * g[i][j];
        } else {
            for (let j = 0; j < g.length; j++)
                sumSq += g[j] * g[j];
        }
        const norm = Math.sqrt(sumSq);
        if (norm <= GRAD_NORM_LIMIT) return g;
        const scale = GRAD_NORM_LIMIT / norm;
        if (g[0] instanceof Float64Array || Array.isArray(g[0])) {
            const c = Array.from({ length: g.length }, () => new Float64Array(g[0].length));
            for (let i = 0; i < g.length; i++)
                for (let j = 0; j < g[0].length; j++)
                    c[i][j] = g[i][j] * scale;
            return c;
        } else {
            const c = new Float64Array(g.length);
            for (let j = 0; j < g.length; j++)
                c[j] = g[j] * scale;
            return c;
        }
    }

    _updateWeights(w, dw, b, db, lr) {
        dw = this._clipNorm(dw);
        db = this._clipNorm(db);
        for (let i = 0; i < w.length; i++)
            for (let j = 0; j < w[0].length; j++)
                w[i][j] -= lr * dw[i][j];
        for (let j = 0; j < b.length; j++)
            b[j] -= lr * db[j];
    }

    getAction(board, validActions, epsilon = 0) {
        if (!validActions || validActions.length === 0) return 0;
        if (Math.random() < epsilon)
            return validActions[Math.floor(Math.random() * validActions.length)];
        const q = this.forward(board);
        let bestA = validActions[0], bestQ = -Infinity;
        for (const a of validActions) {
            if (q[a] > bestQ) { bestQ = q[a]; bestA = a; }
        }
        return bestA;
    }

    toJSON() {
        const to2D = (m) => m.map(r => Array.from(r));
        const to1D = (a) => Array.from(a);
        return {
            hiddenSize: this.hiddenSize,
            w1: to2D(this.w1), b1: to1D(this.b1),
            w2: to2D(this.w2), b2: to1D(this.b2),
            w3: to2D(this.w3), b3: to1D(this.b3),
        };
    }

    static fromJSON(obj) {
        const net = new QNetwork(obj.hiddenSize);
        const toF64_2D = (arr) => arr.map(r => new Float64Array(r));
        const toF64_1D = (arr) => new Float64Array(arr);
        net.w1 = toF64_2D(obj.w1); net.b1 = toF64_1D(obj.b1);
        net.w2 = toF64_2D(obj.w2); net.b2 = toF64_1D(obj.b2);
        net.w3 = toF64_2D(obj.w3); net.b3 = toF64_1D(obj.b3);
        return net;
    }
}
