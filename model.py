import numpy as np

NUM_CHANNELS = 16  # 원핫 채널 수: 0, 2^1, 2^2, ..., 2^15
INPUT_SIZE = 4 * 4 * NUM_CHANNELS  # 256
GRAD_NORM_LIMIT = 1.0
HUBER_DELTA = 1.0


class QNetwork:
    """2048용 Q-Network (NumPy 구현)"""

    def __init__(self, hidden_size: int = 128):
        self.hidden_size = hidden_size

        # Xavier 초기화
        self.w1 = np.random.randn(INPUT_SIZE, hidden_size) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.w3 = np.random.randn(hidden_size, 4) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(4)

        # 캐시 (역전파용)
        self._cache = {}

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """보드 전처리: 원핫 인코딩 (각 셀 → 16채널)"""
        x = x.reshape(-1, 16).astype(np.int32)
        batch = x.shape[0]
        onehot = np.zeros((batch, INPUT_SIZE), dtype=np.float32)
        for b in range(batch):
            for i in range(16):
                v = x[b, i]
                if v == 0:
                    ch = 0
                else:
                    ch = int(np.log2(v))
                    if ch >= NUM_CHANNELS:
                        ch = NUM_CHANNELS - 1
                onehot[b, i * NUM_CHANNELS + ch] = 1.0
        return onehot

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

        x = self._preprocess(state)

        # Layer 1
        z1 = x @ self.w1 + self.b1
        a1 = self._relu(z1)

        # Layer 2
        z2 = a1 @ self.w2 + self.b2
        a2 = self._relu(z2)

        # Layer 3 (출력)
        z3 = a2 @ self.w3 + self.b3

        # 캐시 저장
        self._cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}

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
            loss = 0.5 * error ** 2
            dloss = error
        else:
            loss = HUBER_DELTA * (abs_error - 0.5 * HUBER_DELTA)
            dloss = HUBER_DELTA * np.sign(error)

        # 출력층 기울기
        dz3 = np.zeros_like(q_values)
        dz3[0, action] = dloss

        # Layer 3
        dw3 = cache["a2"].T @ dz3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.w3.T

        # Layer 2
        dz2 = self._relu_backward(da2, cache["z2"])
        dw2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.w2.T

        # Layer 1
        dz1 = self._relu_backward(da1, cache["z1"])
        dw1 = cache["x"].T @ dz1
        db1 = dz1.sum(axis=0)

        # Gradient clipping for weights
        dw3, dw2, dw1, db3, db2, db1 = self._clip_gradients(
            dw3, dw2, dw1, db3, db2, db1
        )

        # 가중치 업데이트
        self.w3 -= lr * dw3
        self.b3 -= lr * db3
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

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
            path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3
        )

    def load(self, path: str):
        """모델 로드"""
        data = np.load(path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w3, self.b3 = data["w3"], data["b3"]


# 테스트
if __name__ == "__main__":
    model = QNetwork(hidden_size=64)

    # 더미 상태
    state = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [8, 16, 128, 64], [8, 16, 128, 64]])

    # 순전파
    q_values = model.forward(state)
    print(f"Q values: {q_values}")

    # 행동 선택
    action = model.get_action(state, epsilon=0.0)
    print(f"Selected action: {action}")

    # 역전파
    loss = model.backward(action=1, target=100.0, lr=0.01)
    print(f"Loss: {loss}")
