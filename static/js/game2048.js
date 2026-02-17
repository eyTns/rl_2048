// ============================================================
// Game2048 JS 클래스
// ============================================================

const SPAWN_4_RATE = 0.1;
const ACTION_UP = 0, ACTION_DOWN = 1, ACTION_LEFT = 2, ACTION_RIGHT = 3;
const ACTION_NAMES = ['↑', '↓', '←', '→'];
const ROTATION_MAP = { [ACTION_UP]: 1, [ACTION_DOWN]: 3, [ACTION_LEFT]: 0, [ACTION_RIGHT]: 2 };

const CURRICULUM_TILES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

class Game2048 {
    constructor(curriculumMode = false) {
        this.board = null;
        this.score = 0;
        this.done = false;
        this.curriculumMode = curriculumMode;
    }

    reset(curriculumMode = null) {
        const mode = curriculumMode !== null ? curriculumMode : this.curriculumMode;
        this.board = Array.from({ length: 4 }, () => new Array(4).fill(0));
        this.score = 0;
        this.done = false;
        if (mode) {
            this._resetCurriculum();
        } else {
            this._spawnTile();
            this._spawnTile();
        }
        return this.getState();
    }

    _resetCurriculum() {
        // 2^1 ~ 2^10 중 4개 복원추출 → 내림차순 정렬
        const tiles = [];
        for (let i = 0; i < 4; i++) {
            tiles.push(CURRICULUM_TILES[Math.floor(Math.random() * CURRICULUM_TILES.length)]);
        }
        tiles.sort((a, b) => b - a);
        this.board[3] = tiles;
        // 우상단에 2 고정
        this.board[0][3] = 2;
    }

    getState() {
        return this.board.map(r => [...r]);
    }

    step(action) {
        if (this.done) return { state: this.getState(), reward: 0, done: true };
        const oldBoard = JSON.stringify(this.board);
        const reward = this._move(action);
        const validMove = JSON.stringify(this.board) !== oldBoard;
        if (validMove) {
            this._spawnTile();
            if (!this._canMove()) this.done = true;
        }
        this.score += reward;
        return { state: this.getState(), reward, done: this.done };
    }

    getValidActions() {
        const valid = [];
        // left=2, right=3, up=0, down=1
        if (this._canMoveDirection(0)) valid.push(0); // up
        if (this._canMoveDirection(1)) valid.push(1); // down
        if (this._canMoveDirection(2)) valid.push(2); // left
        if (this._canMoveDirection(3)) valid.push(3); // right
        return valid;
    }

    _canMoveRowLeft(row) {
        for (let j = 0; j < 4; j++) {
            if (row[j] === 0) {
                for (let k = j + 1; k < 4; k++) {
                    if (row[k] !== 0) return true;
                }
            } else if (j + 1 < 4 && row[j] === row[j + 1]) {
                return true;
            }
        }
        return false;
    }

    _canMoveDirection(action) {
        const k = ROTATION_MAP[action];
        const rotated = this._rotate(this.board, k);
        for (let i = 0; i < 4; i++) {
            if (this._canMoveRowLeft(rotated[i])) return true;
        }
        return false;
    }

    getMaxTile() {
        let max = 0;
        for (const row of this.board) for (const v of row) if (v > max) max = v;
        return max;
    }

    _spawnTile() {
        const empty = [];
        for (let i = 0; i < 4; i++)
            for (let j = 0; j < 4; j++)
                if (this.board[i][j] === 0) empty.push([i, j]);
        if (empty.length === 0) return;
        const [r, c] = empty[Math.floor(Math.random() * empty.length)];
        this.board[r][c] = Math.random() < SPAWN_4_RATE ? 4 : 2;
    }

    _move(action) {
        const k = ROTATION_MAP[action];
        let rotated = this._rotate(this.board, k);
        let reward = 0;
        for (let i = 0; i < 4; i++) {
            const [merged, r] = this._mergeRow(rotated[i]);
            rotated[i] = merged;
            reward += r;
        }
        this.board = this._rotate(rotated, (4 - k) % 4);
        return reward;
    }

    _mergeRow(row) {
        const nonZero = row.filter(x => x !== 0);
        const merged = [];
        let reward = 0;
        let i = 0;
        while (i < nonZero.length) {
            if (i + 1 < nonZero.length && nonZero[i] === nonZero[i + 1]) {
                const val = nonZero[i] * 2;
                merged.push(val);
                reward += val;
                i += 2;
            } else {
                merged.push(nonZero[i]);
                i++;
            }
        }
        while (merged.length < 4) merged.push(0);
        return [merged, reward];
    }

    _rotate(b, k) {
        let r = b.map(row => [...row]);
        for (let t = 0; t < k; t++) {
            const n = r.length;
            const rot = Array.from({ length: n }, () => new Array(n).fill(0));
            for (let x = 0; x < n; x++)
                for (let y = 0; y < n; y++)
                    rot[n - 1 - y][x] = r[x][y];  // 반시계 (np.rot90 일치)
            r = rot;
        }
        return r;
    }

    _canMove() {
        for (let i = 0; i < 4; i++)
            for (let j = 0; j < 4; j++) {
                if (this.board[i][j] === 0) return true;
                if (j < 3 && this.board[i][j] === this.board[i][j + 1]) return true;
                if (i < 3 && this.board[i][j] === this.board[i + 1][j]) return true;
            }
        return false;
    }
}
