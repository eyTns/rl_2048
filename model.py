import numpy as np

NUM_CHANNELS = 16  # 원핫 채널 수: 0, 2^1, 2^2, ..., 2^15
GRAD_NORM_LIMIT = 1.0
HUBER_DELTA = 1.0

# Conv 필터 수
_C_CONV = 16


def _im2col(x_padded, kh, kw, H_out, W_out):
    """컨볼루션을 위한 패치 추출.

    Args:
        x_padded: (batch, H_pad, W_pad, C_in) 패딩된 입력
        kh, kw: 커널 높이/너비
        H_out, W_out: 출력 공간 크기

    Returns:
        (batch * H_out * W_out, kh * kw * C_in) 행렬
    """
    batch, _, _, C_in = x_padded.shape
    col = np.zeros((batch, H_out, W_out, kh, kw, C_in), dtype=x_padded.dtype)
    for i in range(kh):
        for j in range(kw):
            col[:, :, :, i, j, :] = x_padded[:, i : i + H_out, j : j + W_out, :]
    return col.reshape(batch * H_out * W_out, kh * kw * C_in)


def _col2im(dcol, x_padded_shape, kh, kw, H_out, W_out):
    """그래디언트를 패딩된 입력 형태로 복원.

    Args:
        dcol: (batch * H_out * W_out, kh * kw * C_in)
        x_padded_shape: 패딩된 입력의 원래 shape
        kh, kw: 커널 높이/너비
        H_out, W_out: 출력 공간 크기

    Returns:
        패딩된 입력과 같은 shape의 그래디언트
    """
    batch = x_padded_shape[0]
    C_in = x_padded_shape[3]
    dcol = dcol.reshape(batch, H_out, W_out, kh, kw, C_in)
    dx = np.zeros(x_padded_shape, dtype=dcol.dtype)
    for i in range(kh):
        for j in range(kw):
            dx[:, i : i + H_out, j : j + W_out, :] += dcol[:, :, :, i, j, :]
    return dx


class QNetwork:
    """2048용 Q-Network — Conv Branch 아키텍처 (NumPy 구현)

    구조:
        Input (4,4,16)
          ├─ Branch A: Conv2D(16, (1,4), 'same') → ReLU  [수평]
          └─ Branch B: Conv2D(16, (4,1), 'same') → ReLU  [수직]
          ↓ Concatenate → (4,4,32) → Flatten → 512
        Dense 1: 512 → 256, ReLU
        Dense 2: 256 → 64, ReLU
        Output:  64 → 4, Linear
    """

    def __init__(self, hidden_size: int = 256):
        self.hidden_size = hidden_size

        C_in = NUM_CHANNELS  # 16
        dense2_size = 64
        concat_size = 4 * 4 * (_C_CONV * 2)  # 512

        # Branch A: Conv2D — 커널 (1,4), 수평 행 전체 커버
        # 'same' 패딩: pad_h=(0,0), pad_w=(1,2)
        self.conv_a_w = np.random.randn(1, 4, C_in, _C_CONV) * np.sqrt(
            2.0 / (1 * 4 * C_in)
        )
        self.conv_a_b = np.zeros(_C_CONV)
        self._pad_a = ((0, 0), (1, 2))

        # Branch B: Conv2D — 커널 (4,1), 수직 열 전체 커버
        # 'same' 패딩: pad_h=(1,2), pad_w=(0,0)
        self.conv_b_w = np.random.randn(4, 1, C_in, _C_CONV) * np.sqrt(
            2.0 / (4 * 1 * C_in)
        )
        self.conv_b_b = np.zeros(_C_CONV)
        self._pad_b = ((1, 2), (0, 0))

        # Dense 1: 512 → hidden_size (256)
        self.w1 = np.random.randn(concat_size, hidden_size) * np.sqrt(
            2.0 / concat_size
        )
        self.b1 = np.zeros(hidden_size)

        # Dense 2: hidden_size (256) → 64
        self.w2 = np.random.randn(hidden_size, dense2_size) * np.sqrt(
            2.0 / hidden_size
        )
        self.b2 = np.zeros(dense2_size)

        # Output: 64 → 4
        self.w3 = np.random.randn(dense2_size, 4) * np.sqrt(2.0 / dense2_size)
        self.b3 = np.zeros(4)

        # 캐시 (역전파용)
        self._cache = {}

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """보드 전처리: 원핫 인코딩 → (batch, 4, 4, 16)"""
        x = x.reshape(-1, 4, 4).astype(np.int32)
        batch = x.shape[0]
        onehot = np.zeros((batch, 4, 4, NUM_CHANNELS), dtype=np.float32)
        for b in range(batch):
            for r in range(4):
                for c in range(4):
                    v = x[b, r, c]
                    if v == 0:
                        ch = 0
                    else:
                        ch = int(np.log2(v))
                        if ch >= NUM_CHANNELS:
                            ch = NUM_CHANNELS - 1
                    onehot[b, r, c, ch] = 1.0
        return onehot

    def _conv2d_forward(self, x, w, b, pad):
        """Conv2D 순전파 ('same' 패딩).

        Args:
            x: (batch, H, W, C_in)
            w: (kh, kw, C_in, C_out)
            b: (C_out,)
            pad: ((pad_top, pad_bottom), (pad_left, pad_right))

        Returns:
            out: (batch, H, W, C_out)
            col: im2col 결과 (역전파용 캐시)
            x_padded_shape: 패딩된 입력 shape (역전파용 캐시)
        """
        kh, kw, _, C_out = w.shape
        batch, H, W, _ = x.shape

        x_padded = np.pad(x, ((0, 0), pad[0], pad[1], (0, 0)), mode="constant")
        col = _im2col(x_padded, kh, kw, H, W)
        w_col = w.reshape(-1, C_out)

        out = (col @ w_col + b).reshape(batch, H, W, C_out)
        return out, col, x_padded.shape

    def _conv2d_backward(self, dout, col, x_padded_shape, w, pad):
        """Conv2D 역전파.

        Args:
            dout: (batch, H, W, C_out) 출력 그래디언트
            col: 순전파에서 캐시된 im2col 결과
            x_padded_shape: 패딩된 입력 shape
            w: (kh, kw, C_in, C_out) 가중치
            pad: ((pad_top, pad_bottom), (pad_left, pad_right))

        Returns:
            dx: 입력 그래디언트 (패딩 제거 후)
            dw: 가중치 그래디언트
            db: 바이어스 그래디언트
        """
        kh, kw, _, C_out = w.shape
        _, H, W, _ = dout.shape

        dout_col = dout.reshape(-1, C_out)
        w_col = w.reshape(-1, C_out)

        dw = (col.T @ dout_col).reshape(w.shape)
        db = dout_col.sum(axis=0)

        dcol = dout_col @ w_col.T
        dx_padded = _col2im(dcol, x_padded_shape, kh, kw, H, W)

        # 패딩 제거
        pt, pb = pad[0]
        pl, pr = pad[1]
        h_start = pt
        h_end = dx_padded.shape[1] - pb if pb > 0 else dx_padded.shape[1]
        w_start = pl
        w_end = dx_padded.shape[2] - pr if pr > 0 else dx_padded.shape[2]
        dx = dx_padded[:, h_start:h_end, w_start:w_end, :]

        return dx, dw, db

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        return grad * (x > 0)

    def _clip_gradients(self, *grads):
        """그래디언트 노름 클리핑 (방향 보존)"""
        clipped = []
        for g in grads:
            norm = np.linalg.norm(g)
            if norm > GRAD_NORM_LIMIT:
                g = g * (GRAD_NORM_LIMIT / norm)
            clipped.append(g)
        return clipped

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        순전파

        Args:
            state: (4, 4) 또는 (batch, 4, 4) 형태
        Returns:
            (4,) 또는 (batch, 4) Q값
        """
        single = state.ndim == 2
        if single:
            state = state[np.newaxis, ...]

        x = self._preprocess(state)  # (batch, 4, 4, 16)

        # Branch A: Conv(1,4) → ReLU
        conv_a_z, col_a, xps_a = self._conv2d_forward(
            x, self.conv_a_w, self.conv_a_b, self._pad_a
        )
        conv_a_a = self._relu(conv_a_z)

        # Branch B: Conv(4,1) → ReLU
        conv_b_z, col_b, xps_b = self._conv2d_forward(
            x, self.conv_b_w, self.conv_b_b, self._pad_b
        )
        conv_b_a = self._relu(conv_b_z)

        # Concatenate → Flatten
        merged = np.concatenate([conv_a_a, conv_b_a], axis=-1)  # (batch,4,4,32)
        flat = merged.reshape(state.shape[0], -1)  # (batch, 512)

        # Dense 1
        z1 = flat @ self.w1 + self.b1
        a1 = self._relu(z1)

        # Dense 2
        z2 = a1 @ self.w2 + self.b2
        a2 = self._relu(z2)

        # Output
        z3 = a2 @ self.w3 + self.b3

        self._cache = {
            "x": x,
            "conv_a_z": conv_a_z,
            "col_a": col_a,
            "xps_a": xps_a,
            "conv_b_z": conv_b_z,
            "col_b": col_b,
            "xps_b": xps_b,
            "merged": merged,
            "flat": flat,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "z3": z3,
        }

        return z3[0] if single else z3

    def backward(self, action: int, target: float, lr: float = 0.001) -> float:
        """
        역전파 및 가중치 업데이트

        Args:
            action: 선택된 행동 (0-3)
            target: 타겟 Q값
            lr: 학습률
        Returns:
            손실값
        """
        cache = self._cache
        q_values = cache["z3"]  # (1, 4)

        # NaN 체크
        if np.any(np.isnan(q_values)):
            return 0.0

        # Huber Loss (δ=1.0): 큰 오차에 둔감하여 발산 방지
        q_value = q_values[0, action]
        error = q_value - target
        abs_error = abs(error)
        if abs_error <= HUBER_DELTA:
            loss = 0.5 * error**2
            dloss = error
        else:
            loss = HUBER_DELTA * (abs_error - 0.5 * HUBER_DELTA)
            dloss = HUBER_DELTA * np.sign(error)

        # 출력층 기울기
        dz3 = np.zeros_like(q_values)
        dz3[0, action] = dloss

        # Output: z3 = a2 @ w3 + b3
        dw3 = cache["a2"].T @ dz3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.w3.T

        # Dense 2 + ReLU
        dz2 = self._relu_backward(da2, cache["z2"])
        dw2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.w2.T

        # Dense 1 + ReLU
        dz1 = self._relu_backward(da1, cache["z1"])
        dw1 = cache["flat"].T @ dz1
        db1 = dz1.sum(axis=0)
        dflat = dz1 @ self.w1.T

        # Unflatten → (batch, 4, 4, 32)
        dmerged = dflat.reshape(cache["merged"].shape)

        # Concatenate 분할: 앞 16채널 → Branch A, 뒤 16채널 → Branch B
        dconv_a_a = dmerged[:, :, :, :_C_CONV]
        dconv_b_a = dmerged[:, :, :, _C_CONV:]

        # Branch A: ReLU → Conv 역전파
        dconv_a_z = self._relu_backward(dconv_a_a, cache["conv_a_z"])
        _, dconv_a_w, dconv_a_b = self._conv2d_backward(
            dconv_a_z, cache["col_a"], cache["xps_a"], self.conv_a_w, self._pad_a
        )

        # Branch B: ReLU → Conv 역전파
        dconv_b_z = self._relu_backward(dconv_b_a, cache["conv_b_z"])
        _, dconv_b_w, dconv_b_b = self._conv2d_backward(
            dconv_b_z, cache["col_b"], cache["xps_b"], self.conv_b_w, self._pad_b
        )

        # Gradient clipping
        (
            dw3,
            dw2,
            dw1,
            db3,
            db2,
            db1,
            dconv_a_w,
            dconv_a_b,
            dconv_b_w,
            dconv_b_b,
        ) = self._clip_gradients(
            dw3, dw2, dw1, db3, db2, db1, dconv_a_w, dconv_a_b, dconv_b_w, dconv_b_b
        )

        # 가중치 업데이트
        self.w3 -= lr * dw3
        self.b3 -= lr * db3
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.conv_a_w -= lr * dconv_a_w
        self.conv_a_b -= lr * dconv_a_b
        self.conv_b_w -= lr * dconv_b_w
        self.conv_b_b -= lr * dconv_b_b

        return loss

    def get_action(
        self,
        state: np.ndarray,
        valid_actions: list[int] | None = None,
        epsilon: float = 0.0,
    ) -> int:
        """
        epsilon-greedy 행동 선택

        Args:
            state: (4, 4) numpy array
            valid_actions: 유효한 행동 리스트
            epsilon: 탐험 확률
        Returns:
            선택된 행동 (0-3)
        """
        if valid_actions is None:
            valid_actions = [0, 1, 2, 3]

        if len(valid_actions) == 0:
            return 0

        # 탐험
        if np.random.random() < epsilon:
            return np.random.choice(valid_actions)

        # 활용
        q_values = self.forward(state)

        # 유효하지 않은 행동 마스킹
        masked_q = np.full(4, -np.inf)
        for a in valid_actions:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    def save(self, path: str):
        """모델 저장"""
        np.savez(
            path,
            conv_a_w=self.conv_a_w,
            conv_a_b=self.conv_a_b,
            conv_b_w=self.conv_b_w,
            conv_b_b=self.conv_b_b,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
        )

    def load(self, path: str):
        """모델 로드"""
        data = np.load(path)
        self.conv_a_w = data["conv_a_w"]
        self.conv_a_b = data["conv_a_b"]
        self.conv_b_w = data["conv_b_w"]
        self.conv_b_b = data["conv_b_b"]
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w3, self.b3 = data["w3"], data["b3"]


# 테스트
if __name__ == "__main__":
    model = QNetwork(hidden_size=256)

    # 더미 상태
    state = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [8, 16, 128, 64], [8, 16, 128, 64]])

    # 순전파
    q_values = model.forward(state)
    print(f"Q values: {q_values}")
    print(f"Q values shape: {q_values.shape}")

    # 행동 선택
    action = model.get_action(state, epsilon=0.0)
    print(f"Selected action: {action}")

    # 역전파
    loss = model.backward(action=1, target=100.0, lr=0.01)
    print(f"Loss: {loss}")

    # 파라미터 수 출력
    total = (
        model.conv_a_w.size
        + model.conv_a_b.size
        + model.conv_b_w.size
        + model.conv_b_b.size
        + model.w1.size
        + model.b1.size
        + model.w2.size
        + model.b2.size
        + model.w3.size
        + model.b3.size
    )
    print(f"Total parameters: {total:,}")
