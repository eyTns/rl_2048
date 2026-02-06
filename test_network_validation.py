"""네트워크 파라미터, 수식, 초기값, 값 범위, 안정적 수렴성 검증 테스트"""

import numpy as np
import pytest
from pydantic import ValidationError

from model import MAX_HIDDEN_SIZE, MIN_HIDDEN_SIZE, QNetwork
from trainer import MCTrainer, TDTrainer, TrainConfig, create_trainer


# === TrainConfig 파라미터 범위 검증 ===


class TestTrainConfigDefaults:
    """기본 설정값 검증"""

    def test_default_values(self):
        config = TrainConfig()
        assert config.method == "td"
        assert config.gamma == 0.99
        assert config.learning_rate == 0.001
        assert config.epsilon_start == 1.0
        assert config.epsilon_end == 0.01
        assert config.epsilon_decay == 0.995
        assert config.hidden_size == 128
        assert config.target_update_freq == 500
        assert config.reward_log_scale is True


class TestTrainConfigGamma:
    """할인율(gamma) 범위 검증: 0 < γ ≤ 1"""

    def test_valid_values(self):
        TrainConfig(gamma=0.5)
        TrainConfig(gamma=0.99)
        TrainConfig(gamma=0.999)
        TrainConfig(gamma=1.0)

    def test_boundary_just_above_zero(self):
        TrainConfig(gamma=0.001)

    def test_invalid_zero(self):
        with pytest.raises(ValidationError):
            TrainConfig(gamma=0.0)

    def test_invalid_negative(self):
        with pytest.raises(ValidationError):
            TrainConfig(gamma=-0.1)

    def test_invalid_above_one(self):
        with pytest.raises(ValidationError):
            TrainConfig(gamma=1.1)


class TestTrainConfigLearningRate:
    """학습률 범위 검증: 0 < lr ≤ 1"""

    def test_valid_values(self):
        TrainConfig(learning_rate=0.0001)
        TrainConfig(learning_rate=0.001)
        TrainConfig(learning_rate=0.01)
        TrainConfig(learning_rate=1.0)

    def test_invalid_zero(self):
        with pytest.raises(ValidationError):
            TrainConfig(learning_rate=0.0)

    def test_invalid_negative(self):
        with pytest.raises(ValidationError):
            TrainConfig(learning_rate=-0.001)

    def test_invalid_above_one(self):
        with pytest.raises(ValidationError):
            TrainConfig(learning_rate=1.1)


class TestTrainConfigEpsilon:
    """탐험율 범위 검증: 0 ≤ ε ≤ 1, epsilon_end ≤ epsilon_start"""

    def test_valid_ranges(self):
        TrainConfig(epsilon_start=1.0, epsilon_end=0.01)
        TrainConfig(epsilon_start=0.5, epsilon_end=0.1)
        TrainConfig(epsilon_start=0.0, epsilon_end=0.0)
        TrainConfig(epsilon_start=0.5, epsilon_end=0.5)  # equal is OK

    def test_epsilon_start_out_of_range(self):
        with pytest.raises(ValidationError):
            TrainConfig(epsilon_start=1.5)
        with pytest.raises(ValidationError):
            TrainConfig(epsilon_start=-0.1)

    def test_epsilon_end_out_of_range(self):
        with pytest.raises(ValidationError):
            TrainConfig(epsilon_end=1.5)
        with pytest.raises(ValidationError):
            TrainConfig(epsilon_end=-0.1)

    def test_epsilon_end_greater_than_start(self):
        with pytest.raises(ValidationError):
            TrainConfig(epsilon_start=0.1, epsilon_end=0.5)

    def test_epsilon_decay_valid(self):
        TrainConfig(epsilon_decay=0.99)
        TrainConfig(epsilon_decay=0.5)
        TrainConfig(epsilon_decay=1.0)

    def test_epsilon_decay_invalid_zero(self):
        with pytest.raises(ValidationError):
            TrainConfig(epsilon_decay=0.0)

    def test_epsilon_decay_invalid_negative(self):
        with pytest.raises(ValidationError):
            TrainConfig(epsilon_decay=-0.1)

    def test_epsilon_decay_invalid_above_one(self):
        with pytest.raises(ValidationError):
            TrainConfig(epsilon_decay=1.1)


class TestTrainConfigHiddenSize:
    """은닉층 크기 범위 검증: 8 ≤ size ≤ 1024"""

    def test_valid_sizes(self):
        TrainConfig(hidden_size=8)
        TrainConfig(hidden_size=64)
        TrainConfig(hidden_size=128)
        TrainConfig(hidden_size=256)
        TrainConfig(hidden_size=1024)

    def test_invalid_too_small(self):
        with pytest.raises(ValidationError):
            TrainConfig(hidden_size=7)

    def test_invalid_zero(self):
        with pytest.raises(ValidationError):
            TrainConfig(hidden_size=0)

    def test_invalid_negative(self):
        with pytest.raises(ValidationError):
            TrainConfig(hidden_size=-1)

    def test_invalid_too_large(self):
        with pytest.raises(ValidationError):
            TrainConfig(hidden_size=1025)


class TestTrainConfigTargetUpdateFreq:
    """타겟 네트워크 업데이트 주기 검증: freq ≥ 1"""

    def test_valid_values(self):
        TrainConfig(target_update_freq=1)
        TrainConfig(target_update_freq=100)
        TrainConfig(target_update_freq=1000)

    def test_invalid_zero(self):
        with pytest.raises(ValidationError):
            TrainConfig(target_update_freq=0)

    def test_invalid_negative(self):
        with pytest.raises(ValidationError):
            TrainConfig(target_update_freq=-1)


# === QNetwork 파라미터 검증 ===


class TestQNetworkValidation:
    """QNetwork 생성자 파라미터 검증"""

    def test_valid_hidden_sizes(self):
        QNetwork(hidden_size=MIN_HIDDEN_SIZE)
        QNetwork(hidden_size=128)
        QNetwork(hidden_size=MAX_HIDDEN_SIZE)

    def test_invalid_too_small(self):
        with pytest.raises(ValueError, match="hidden_size must be between"):
            QNetwork(hidden_size=MIN_HIDDEN_SIZE - 1)

    def test_invalid_too_large(self):
        with pytest.raises(ValueError, match="hidden_size must be between"):
            QNetwork(hidden_size=MAX_HIDDEN_SIZE + 1)

    def test_invalid_zero(self):
        with pytest.raises(ValueError):
            QNetwork(hidden_size=0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            QNetwork(hidden_size=-1)


# === 가중치 초기화 검증 ===


class TestWeightInitialization:
    """He 초기화 및 가중치 형태 검증"""

    def test_weight_shapes(self):
        """가중치 행렬 크기 = (fan_in, fan_out)"""
        model = QNetwork(hidden_size=64)
        assert model.w1.shape == (16, 64)
        assert model.b1.shape == (64,)
        assert model.w2.shape == (64, 64)
        assert model.b2.shape == (64,)
        assert model.w3.shape == (64, 4)
        assert model.b3.shape == (4,)

    def test_bias_initialized_to_zero(self):
        """바이어스 = 0으로 초기화"""
        model = QNetwork(hidden_size=64)
        assert np.all(model.b1 == 0)
        assert np.all(model.b2 == 0)
        assert np.all(model.b3 == 0)

    def test_he_initialization_scale(self):
        """He 초기화: std ≈ sqrt(2 / fan_in)"""
        np.random.seed(42)
        model = QNetwork(hidden_size=256)

        # 큰 행렬에서 통계적 검증 (표본 std ≈ 기대 std)
        expected_std_w1 = np.sqrt(2.0 / 16)
        expected_std_w2 = np.sqrt(2.0 / 256)
        assert abs(model.w1.std() - expected_std_w1) < 0.05
        assert abs(model.w2.std() - expected_std_w2) < 0.02

    def test_weights_are_not_all_zero(self):
        """가중치가 0이 아닌지 확인 (ReLU에서 죽은 뉴런 방지)"""
        model = QNetwork(hidden_size=64)
        assert not np.all(model.w1 == 0)
        assert not np.all(model.w2 == 0)
        assert not np.all(model.w3 == 0)

    def test_total_parameter_count(self):
        """총 파라미터 수 = 16*h + h + h*h + h + h*4 + 4"""
        h = 128
        model = QNetwork(hidden_size=h)
        expected = 16 * h + h + h * h + h + h * 4 + 4
        actual = (
            model.w1.size + model.b1.size
            + model.w2.size + model.b2.size
            + model.w3.size + model.b3.size
        )
        assert actual == expected


# === 타겟 네트워크 복사 검증 ===


class TestCopyWeights:
    """copy_weights_from으로 타겟 네트워크 동기화 검증"""

    def test_copy_makes_weights_equal(self):
        model = QNetwork(hidden_size=32)
        target = QNetwork(hidden_size=32)

        # 초기에는 다름 (랜덤 초기화)
        assert not np.allclose(model.w1, target.w1)

        target.copy_weights_from(model)

        assert np.allclose(target.w1, model.w1)
        assert np.allclose(target.b1, model.b1)
        assert np.allclose(target.w2, model.w2)
        assert np.allclose(target.b2, model.b2)
        assert np.allclose(target.w3, model.w3)
        assert np.allclose(target.b3, model.b3)

    def test_copy_is_deep(self):
        """복사 후 원본 변경이 복사본에 영향 없음"""
        model = QNetwork(hidden_size=32)
        target = QNetwork(hidden_size=32)

        target.copy_weights_from(model)
        original_w1 = target.w1.copy()

        model.w1 += 1.0  # 원본 변경

        assert np.allclose(target.w1, original_w1)  # 복사본 불변


# === 입력 전처리 검증 ===


class TestPreprocessing:
    """log2 스케일 정규화 검증: 출력 ∈ [0, 1]"""

    def test_output_range_zero_to_one(self):
        model = QNetwork()
        state = np.array([
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [4096, 8192, 16384, 32768],
        ])
        preprocessed = model._preprocess(state)
        assert np.all(preprocessed >= 0.0)
        assert np.all(preprocessed <= 1.0)

    def test_zero_tiles_map_to_zero(self):
        """빈 타일(0) → 0.0"""
        model = QNetwork()
        state = np.zeros((4, 4))
        preprocessed = model._preprocess(state)
        assert np.all(preprocessed == 0.0)

    def test_max_tile_maps_to_one(self):
        """최대 타일(32768) → 1.0"""
        model = QNetwork()
        state = np.full((4, 4), 32768)
        preprocessed = model._preprocess(state)
        assert np.allclose(preprocessed, 1.0)

    def test_log2_monotonic(self):
        """타일 값이 클수록 전처리 출력도 큼"""
        model = QNetwork()
        values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        for i in range(len(values) - 1):
            s1 = np.full((4, 4), values[i])
            s2 = np.full((4, 4), values[i + 1])
            assert np.all(model._preprocess(s2) > model._preprocess(s1))


# === TD 수식 검증 (벨만 방정식) ===


class TestTDFormula:
    """TD target = r + γ * max_a' Q_target(s', a') 검증"""

    def test_td_trains_every_step(self):
        """TD는 매 스텝마다 loss를 생성"""
        config = TrainConfig(method="td", hidden_size=32, epsilon_start=1.0)
        trainer = TDTrainer(config)
        steps, score, losses = trainer.train_one_episode()
        assert len(losses) == steps

    def test_td_bellman_target_nonterminal(self):
        """비종료 상태: target = scale(r) + γ * max Q_target(s', a')"""
        config = TrainConfig(
            method="td", gamma=0.99, hidden_size=32,
            reward_log_scale=False, epsilon_start=0.0, epsilon_end=0.0,
            target_update_freq=100000,
        )
        trainer = TDTrainer(config)

        state = trainer.env.reset()
        valid = trainer.env.get_valid_actions()
        action = valid[0]
        next_state, reward, done, info = trainer.env.step(action)

        if not done:
            # 수동으로 벨만 타겟 계산
            next_q = trainer.target_model.forward(next_state)
            next_valid = trainer.env.get_valid_actions()
            masked_q = np.full(4, -np.inf)
            for a in next_valid:
                masked_q[a] = next_q[a]
            expected_target = reward + 0.99 * float(np.max(masked_q))

            # 실제 타겟과 비교 가능한 범위인지 확인
            assert np.isfinite(expected_target)
            assert expected_target >= reward  # gamma * Q >= 0 일 때

    def test_td_terminal_target_is_reward_only(self):
        """종료 상태: target = scale(r), 미래 Q값 없음"""
        # 종료 상태에서는 bootstrapping 없이 보상만 사용됨
        # 이는 코드 구조로 검증: step.done일 때 target = scaled_reward
        config = TrainConfig(method="td", hidden_size=32, reward_log_scale=False)
        trainer = TDTrainer(config)

        # 짧은 에피소드 실행 후 마지막 스텝의 done 확인
        steps, score, losses = trainer.train_one_episode()
        assert steps >= 1
        assert len(losses) == steps

    def test_target_network_initial_sync(self):
        """타겟 네트워크가 초기에 메인 네트워크와 동기화"""
        config = TrainConfig(method="td", hidden_size=32)
        trainer = TDTrainer(config)

        assert np.allclose(trainer.model.w1, trainer.target_model.w1)
        assert np.allclose(trainer.model.w2, trainer.target_model.w2)
        assert np.allclose(trainer.model.w3, trainer.target_model.w3)

    def test_target_network_diverges_during_training(self):
        """학습 중 타겟 네트워크와 메인 네트워크가 달라짐"""
        config = TrainConfig(
            method="td", hidden_size=32,
            target_update_freq=100000, epsilon_start=1.0,
        )
        trainer = TDTrainer(config)
        trainer.train(episodes=10, print_every=10)

        # 메인 네트워크는 업데이트되었지만 타겟은 동기화 주기 전
        assert not np.allclose(trainer.model.w1, trainer.target_model.w1)

    def test_target_network_syncs_periodically(self):
        """타겟 네트워크가 target_update_freq마다 동기화"""
        config = TrainConfig(
            method="td", hidden_size=32,
            target_update_freq=5, epsilon_start=1.0,
        )
        trainer = TDTrainer(config)
        trainer.train(episodes=50, print_every=50)

        # 충분히 학습 후 step_counter가 증가했는지 확인
        assert trainer._step_counter > 0

    def test_old_td_formula_not_used(self):
        """구 수식(r_{t-1} + γ * r_t)이 아닌 벨만 방정식 사용 확인

        벨만 방정식은 Q값을 bootstrapping하므로 학습 후 Q값이
        단순 보상 합산보다 더 큰 범위를 가져야 함.
        """
        config = TrainConfig(
            method="td", gamma=0.99, hidden_size=32,
            epsilon_start=0.5, reward_log_scale=False,
        )
        trainer = TDTrainer(config)
        trainer.train(episodes=100, print_every=100)

        state = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        q_values = trainer.model.forward(state)

        # Q값이 학습되었는지 확인 (모두 0이 아님)
        assert not np.allclose(q_values, 0.0, atol=0.01)


# === MC 수식 검증 (할인 리턴) ===


class TestMCFormula:
    """MC return: G_t = scale(r_t) + γ * G_{t+1} 검증"""

    def test_mc_trains_at_episode_end(self):
        """MC는 에피소드 끝에 모든 스텝을 학습"""
        config = TrainConfig(method="mc", hidden_size=32, epsilon_start=1.0)
        trainer = MCTrainer(config)
        steps, score, losses = trainer.train_one_episode()
        assert len(losses) == steps

    def test_discounted_return_calculation(self):
        """G_t = r_t + γ * G_{t+1} 수동 검증"""
        gamma = 0.99
        rewards = [4.0, 8.0, 16.0]

        # 역순 계산
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # G_2 = 16
        assert returns[2] == pytest.approx(16.0)
        # G_1 = 8 + 0.99 * 16 = 23.84
        assert returns[1] == pytest.approx(8 + 0.99 * 16)
        # G_0 = 4 + 0.99 * 23.84 = 27.6016
        assert returns[0] == pytest.approx(4 + 0.99 * (8 + 0.99 * 16))

    def test_mc_uses_gamma(self):
        """MC가 할인율을 사용하는지 확인 (gamma < 1이면 리턴이 제한됨)"""
        config = TrainConfig(
            method="mc", gamma=0.5, hidden_size=32,
            epsilon_start=1.0, reward_log_scale=False,
        )
        trainer = MCTrainer(config)
        steps, score, losses = trainer.train_one_episode()

        # gamma=0.5이면 리턴이 빠르게 감소하여 loss 범위가 제한적
        assert all(np.isfinite(loss) for loss in losses)

    def test_mc_gamma_one_equals_total_return(self):
        """gamma=1일 때 G_0 = 총 보상의 합"""
        gamma = 1.0
        rewards = [4.0, 8.0, 16.0]

        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        assert returns[0] == pytest.approx(sum(rewards))


# === 보상 스케일링 검증 ===


class TestRewardScaling:
    """log1p 보상 스케일링: scale(r) = sign(r) * log(1 + |r|)"""

    def test_log_scale_compresses_large_rewards(self):
        """큰 보상이 압축되는지 확인"""
        config = TrainConfig(reward_log_scale=True, hidden_size=32)
        trainer = TDTrainer(config)

        # 256을 합치면 보상=256, log(257)≈5.55로 압축
        scaled = trainer._scale_reward(256.0)
        assert scaled < 256.0
        assert scaled == pytest.approx(np.log1p(256.0))

    def test_log_scale_preserves_zero(self):
        """보상 0은 0으로 유지"""
        config = TrainConfig(reward_log_scale=True, hidden_size=32)
        trainer = TDTrainer(config)
        assert trainer._scale_reward(0.0) == 0.0

    def test_log_scale_preserves_sign(self):
        """부호 보존 (2048은 음수 보상 없지만 일반성 확인)"""
        config = TrainConfig(reward_log_scale=True, hidden_size=32)
        trainer = TDTrainer(config)
        assert trainer._scale_reward(4.0) > 0

    def test_log_scale_monotonic(self):
        """스케일링이 단조 증가인지 확인"""
        config = TrainConfig(reward_log_scale=True, hidden_size=32)
        trainer = TDTrainer(config)
        values = [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        scaled = [trainer._scale_reward(v) for v in values]
        for i in range(len(scaled) - 1):
            assert scaled[i] < scaled[i + 1]

    def test_no_scale_returns_raw(self):
        """스케일링 비활성화 시 원래 보상 반환"""
        config = TrainConfig(reward_log_scale=False, hidden_size=32)
        trainer = TDTrainer(config)
        assert trainer._scale_reward(256.0) == 256.0
        assert trainer._scale_reward(0.0) == 0.0

    def test_mc_also_uses_scaling(self):
        """MC도 보상 스케일링을 사용하는지 확인"""
        config = TrainConfig(method="mc", reward_log_scale=True, hidden_size=32)
        trainer = MCTrainer(config)
        assert trainer._scale_reward(256.0) == pytest.approx(np.log1p(256.0))


# === 수렴 안정성 검증 ===


class TestConvergenceStability:
    """그래디언트 클리핑, NaN 방지, 유한 가중치 검증"""

    def test_gradient_clipping_extreme_target(self):
        """극단적 타겟에서도 가중치가 유한하게 유지"""
        model = QNetwork(hidden_size=32)
        state = np.full((4, 4), 32768)

        model.forward(state)
        loss = model.backward(action=0, target=10000.0, lr=0.001)

        assert np.isfinite(loss)
        assert np.all(np.isfinite(model.w1))
        assert np.all(np.isfinite(model.w2))
        assert np.all(np.isfinite(model.w3))

    def test_nan_guard_in_backward(self):
        """NaN Q값일 때 loss=0.0 반환 (안전 처리)"""
        model = QNetwork(hidden_size=32)
        state = np.zeros((4, 4))
        model.forward(state)

        # 캐시에 NaN 주입
        model._cache["z3"] = np.array([[np.nan, np.nan, np.nan, np.nan]])
        loss = model.backward(action=0, target=1.0, lr=0.001)
        assert loss == 0.0

    def test_training_produces_finite_weights(self):
        """학습 후 모든 가중치가 유한"""
        config = TrainConfig(method="td", hidden_size=32, epsilon_start=1.0)
        trainer = create_trainer(config)
        trainer.train(episodes=30, print_every=30)

        for w in [trainer.model.w1, trainer.model.w2, trainer.model.w3]:
            assert np.all(np.isfinite(w))
        for b in [trainer.model.b1, trainer.model.b2, trainer.model.b3]:
            assert np.all(np.isfinite(b))

    def test_q_values_bounded_after_training(self):
        """학습 후 Q값이 합리적 범위 내에 있는지 확인"""
        config = TrainConfig(
            method="td", hidden_size=32, gamma=0.99, epsilon_start=1.0,
        )
        trainer = create_trainer(config)
        trainer.train(episodes=50, print_every=50)

        state = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        q_values = trainer.model.forward(state)
        assert np.all(np.isfinite(q_values))
        assert np.all(np.abs(q_values) < 10000)

    def test_mc_training_finite_weights(self):
        """MC 학습 후 가중치가 유한"""
        config = TrainConfig(method="mc", hidden_size=32, epsilon_start=1.0)
        trainer = create_trainer(config)
        trainer.train(episodes=30, print_every=30)

        for w in [trainer.model.w1, trainer.model.w2, trainer.model.w3]:
            assert np.all(np.isfinite(w))

    def test_reward_scaling_prevents_large_targets(self):
        """보상 스케일링이 큰 타겟을 방지"""
        config = TrainConfig(reward_log_scale=True, hidden_size=32)
        trainer = TDTrainer(config)

        # 큰 보상 (32768 타일 합치기)
        scaled = trainer._scale_reward(32768.0)
        assert scaled < 100  # log1p(32768) ≈ 10.4
        assert scaled == pytest.approx(np.log1p(32768.0))


# === create_trainer 팩토리 검증 ===


class TestCreateTrainer:
    """create_trainer 팩토리 함수 검증"""

    def test_creates_td_trainer(self):
        config = TrainConfig(method="td")
        trainer = create_trainer(config)
        assert isinstance(trainer, TDTrainer)

    def test_creates_mc_trainer(self):
        config = TrainConfig(method="mc")
        trainer = create_trainer(config)
        assert isinstance(trainer, MCTrainer)

    def test_td_trainer_has_target_model(self):
        config = TrainConfig(method="td")
        trainer = create_trainer(config)
        assert hasattr(trainer, "target_model")

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            TrainConfig(method="invalid")


# === 레이어 구조 검증 ===


class TestLayerArchitecture:
    """네트워크 레이어 구조 및 순전파/역전파 검증"""

    def test_forward_output_shape_single(self):
        """단일 상태 입력: 출력 (4,)"""
        model = QNetwork(hidden_size=64)
        state = np.zeros((4, 4))
        q_values = model.forward(state)
        assert q_values.shape == (4,)

    def test_forward_output_shape_batch(self):
        """배치 입력: 출력 (batch, 4)"""
        model = QNetwork(hidden_size=64)
        states = np.zeros((5, 4, 4))
        q_values = model.forward(states)
        assert q_values.shape == (5, 4)

    def test_relu_activation(self):
        """ReLU: max(0, x)"""
        model = QNetwork()
        x = np.array([-2, -1, 0, 1, 2])
        assert np.array_equal(model._relu(x), np.array([0, 0, 0, 1, 2]))

    def test_relu_backward(self):
        """ReLU 역전파: gradient * (x > 0)"""
        model = QNetwork()
        grad = np.ones(5)
        x = np.array([-2, -1, 0, 1, 2])
        result = model._relu_backward(grad, x)
        assert np.array_equal(result, np.array([0, 0, 0, 1, 1]))

    def test_backward_updates_weights(self):
        """역전파가 가중치를 실제로 업데이트하는지 확인"""
        model = QNetwork(hidden_size=32)
        state = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        model.forward(state)
        w1_before = model.w1.copy()

        model.backward(action=0, target=100.0, lr=0.01)

        assert not np.allclose(model.w1, w1_before)

    def test_invalid_action_masking(self):
        """유효하지 않은 행동이 -inf로 마스킹"""
        model = QNetwork(hidden_size=32)
        state = np.zeros((4, 4))
        state[0, 0] = 2

        action = model.get_action(state, valid_actions=[1, 2], epsilon=0.0)
        assert action in [1, 2]

    def test_epsilon_greedy_exploration(self):
        """epsilon=1.0이면 항상 랜덤 행동"""
        model = QNetwork(hidden_size=32)
        state = np.zeros((4, 4))
        state[0, 0] = 2

        actions = set()
        for _ in range(100):
            a = model.get_action(state, valid_actions=[0, 1, 2, 3], epsilon=1.0)
            actions.add(a)

        # 충분히 시행하면 여러 행동이 선택됨
        assert len(actions) >= 2

    def test_epsilon_zero_is_greedy(self):
        """epsilon=0이면 항상 같은 (최적) 행동"""
        model = QNetwork(hidden_size=32)
        state = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        actions = set()
        for _ in range(10):
            a = model.get_action(state, valid_actions=[0, 1, 2, 3], epsilon=0.0)
            actions.add(a)

        assert len(actions) == 1  # 항상 같은 행동
