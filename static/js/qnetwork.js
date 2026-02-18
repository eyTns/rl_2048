// ============================================================
// QNetwork JS 포팅
// ============================================================

const INPUT_SIZE = 4 * 4;  // 16 (각 셀의 log2 값)
const GRAD_NORM_LIMIT = 1.0;
const HUBER_DELTA = 1.0;

// --- 행렬 유틸 ---
function zeros(rows, cols) {
    if (cols === undefined) return new Float64Array(rows);
    return Array.from({ length: rows }, () => new Float64Array(cols));
}

// In-place matmul: C = A * B (i,p,j 순서 — 캐시 친화적)
function matMulInto(A, B, C) {
    const m = A.length, k = B.length, n = B[0].length;
    for (let i = 0; i < m; i++) C[i].fill(0);
    for (let i = 0; i < m; i++) {
        const Ai = A[i], Ci = C[i];
        for (let p = 0; p < k; p++) {
            const aip = Ai[p], Bp = B[p];
            for (let j = 0; j < n; j++)
                Ci[j] += aip * Bp[j];
        }
    }
}

// In-place bias 덧셈: A[i][j] += b[j]
function addBiasInplace(A, b) {
    for (let i = 0; i < A.length; i++) {
        const Ai = A[i];
        for (let j = 0; j < Ai.length; j++)
            Ai[j] += b[j];
    }
}

// In-place ReLU: R[i][j] = max(0, A[i][j])
function reluInto(A, R) {
    for (let i = 0; i < A.length; i++) {
        const Ai = A[i], Ri = R[i];
        for (let j = 0; j < Ai.length; j++)
            Ri[j] = Ai[j] > 0 ? Ai[j] : 0;
    }
}

// In-place ReLU backward: dz[i][j] = z[i][j] > 0 ? da[i][j] : 0
function reluBackwardInto(da, z, dz) {
    for (let i = 0; i < da.length; i++) {
        const dai = da[i], zi = z[i], dzi = dz[i];
        for (let j = 0; j < dai.length; j++)
            dzi[j] = zi[j] > 0 ? dai[j] : 0;
    }
}

// In-place outer product: C[i][j] = a[i] * b[j]
function outerProductInto(a, b, C) {
    for (let i = 0; i < a.length; i++) {
        const ai = a[i], Ci = C[i];
        for (let j = 0; j < b.length; j++)
            Ci[j] = ai * b[j];
    }
}

// v * W^T (전치 생성 없이): out[j] = Σ_p v[p] * W[j][p]
function vecMatTransposeInto(v, W, out) {
    for (let j = 0; j < W.length; j++) {
        let s = 0;
        const Wj = W[j];
        for (let p = 0; p < v.length; p++) s += v[p] * Wj[p];
        out[j] = s;
    }
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

        this._allocBuffers();
    }

    _allocBuffers() {
        const h = this.hiddenSize;
        this._buf = {
            // forward 버퍼 (backward 캐시 겸용)
            x: zeros(1, INPUT_SIZE),
            z1: zeros(1, h), a1: zeros(1, h),
            z2: zeros(1, h), a2: zeros(1, h),
            z3: zeros(1, 4),
            // backward 활성화 기울기 버퍼
            dz3: zeros(1, 4),
            da2: zeros(1, h), dz2: zeros(1, h),
            da1: zeros(1, h), dz1: zeros(1, h),
            // 가중치 기울기 버퍼
            dw3: zeros(h, 4), db3: zeros(4),
            dw2: zeros(h, h), db2: zeros(h),
            dw1: zeros(INPUT_SIZE, h), db1: zeros(h),
        };
    }

    _randMatrix(rows, cols, scale) {
        const M = Array.from({ length: rows }, () => new Float64Array(cols));
        for (let i = 0; i < rows; i++)
            for (let j = 0; j < cols; j++)
                M[i][j] = randn() * scale;
        return M;
    }

    _preprocessInto(board, x) {
        // board: 4x4 array → x[1][16] 정규화 log2 인코딩 (in-place)
        // 0→0, 2→1/16, 4→2/16, ..., 65536→1
        for (let i = 0; i < 4; i++)
            for (let j = 0; j < 4; j++) {
                const v = board[i][j];
                x[0][i * 4 + j] = v > 0 ? Math.log2(v) / 16 : 0;
            }
    }

    forward(board) {
        const { x, z1, a1, z2, a2, z3 } = this._buf;
        this._preprocessInto(board, x);
        matMulInto(x, this.w1, z1);
        addBiasInplace(z1, this.b1);
        reluInto(z1, a1);
        matMulInto(a1, this.w2, z2);
        addBiasInplace(z2, this.b2);
        reluInto(z2, a2);
        matMulInto(a2, this.w3, z3);
        addBiasInplace(z3, this.b3);
        return [z3[0][0], z3[0][1], z3[0][2], z3[0][3]];
    }

    backward(action, target, lr = 0.001) {
        const { x, z1, a1, z2, a2, z3,
                dz3, da2, dz2, da1, dz1,
                dw3, db3, dw2, db2, dw1, db1 } = this._buf;
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
        dz3[0].fill(0);
        dz3[0][action] = dloss;

        // Layer 3: dw3 = a2^T · dz3, da2 = dz3 · w3^T
        outerProductInto(a2[0], dz3[0], dw3);
        db3.set(dz3[0]);
        vecMatTransposeInto(dz3[0], this.w3, da2[0]);

        // Layer 2 (ReLU backward)
        reluBackwardInto(da2, z2, dz2);
        outerProductInto(a1[0], dz2[0], dw2);
        db2.set(dz2[0]);
        vecMatTransposeInto(dz2[0], this.w2, da1[0]);

        // Layer 1 (ReLU backward)
        reluBackwardInto(da1, z1, dz1);
        outerProductInto(x[0], dz1[0], dw1);
        db1.set(dz1[0]);

        // Clip & update
        this._updateWeights(this.w3, dw3, this.b3, db3, lr);
        this._updateWeights(this.w2, dw2, this.b2, db2, lr);
        this._updateWeights(this.w1, dw1, this.b1, db1, lr);

        return loss;
    }

    _clipNormInplace(g, is2D) {
        let sumSq = 0;
        if (is2D) {
            for (let i = 0; i < g.length; i++) {
                const gi = g[i];
                for (let j = 0; j < gi.length; j++)
                    sumSq += gi[j] * gi[j];
            }
        } else {
            for (let j = 0; j < g.length; j++)
                sumSq += g[j] * g[j];
        }
        const norm = Math.sqrt(sumSq);
        if (norm <= GRAD_NORM_LIMIT) return;
        const scale = GRAD_NORM_LIMIT / norm;
        if (is2D) {
            for (let i = 0; i < g.length; i++) {
                const gi = g[i];
                for (let j = 0; j < gi.length; j++)
                    gi[j] *= scale;
            }
        } else {
            for (let j = 0; j < g.length; j++)
                g[j] *= scale;
        }
    }

    _updateWeights(w, dw, b, db, lr) {
        this._clipNormInplace(dw, true);
        this._clipNormInplace(db, false);
        for (let i = 0; i < w.length; i++) {
            const wi = w[i], dwi = dw[i];
            for (let j = 0; j < wi.length; j++)
                wi[j] -= lr * dwi[j];
        }
        for (let j = 0; j < b.length; j++)
            b[j] -= lr * db[j];
    }

    getAction(board, validActions, epsilon = 0) {
        if (!validActions || validActions.length === 0) return { action: 0, qValues: [0, 0, 0, 0] };
        const q = this.forward(board);
        if (Math.random() < epsilon) {
            return { action: validActions[Math.floor(Math.random() * validActions.length)], qValues: q };
        }
        let bestA = validActions[0], bestQ = -Infinity;
        for (const a of validActions) {
            if (q[a] > bestQ) { bestQ = q[a]; bestA = a; }
        }
        return { action: bestA, qValues: q };
    }

    copyWeightsTo(other) {
        // 가중치 깊은 복사: main → target network 동기화용
        for (let i = 0; i < this.w1.length; i++) other.w1[i].set(this.w1[i]);
        other.b1.set(this.b1);
        for (let i = 0; i < this.w2.length; i++) other.w2[i].set(this.w2[i]);
        other.b2.set(this.b2);
        for (let i = 0; i < this.w3.length; i++) other.w3[i].set(this.w3[i]);
        other.b3.set(this.b3);
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
