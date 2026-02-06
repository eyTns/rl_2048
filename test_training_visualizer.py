"""TrainingVisualizer 테스트"""

import numpy as np
import pytest

from trainer import EpisodeResult, StepInfo, TrainConfig, create_trainer
from training_visualizer import (
    TrainingMetrics,
    TrainingVisualizer,
    _downsample_indices,
    _moving_average,
)


# --- TrainingMetrics 테스트 ---


class TestTrainingMetrics:
    def test_initial_state(self):
        metrics = TrainingMetrics()
        assert metrics.total_steps == 0
        assert metrics.best_score == 0.0
        assert metrics.best_tile == 0
        assert len(metrics.episodes) == 0

    def test_on_step_collects_q_values(self):
        metrics = TrainingMetrics()
        info = StepInfo(
            step_num=0,
            state=np.zeros((4, 4)),
            action=0,
            reward=4.0,
            loss=0.5,
            q_values=np.array([1.0, 2.0, 3.0, 4.0]),
        )
        metrics.on_step(info)

        assert len(metrics._step_q_means) == 1
        assert metrics._step_q_means[0] == pytest.approx(2.5)
        assert metrics._step_q_maxes[0] == pytest.approx(4.0)

    def test_on_step_multiple_steps(self):
        """여러 스텝의 Q값이 누적되는지 확인"""
        metrics = TrainingMetrics()
        for q_max in [4.0, 8.0]:
            metrics.on_step(
                StepInfo(
                    step_num=0,
                    state=np.zeros((4, 4)),
                    action=0,
                    reward=0.0,
                    loss=0.1,
                    q_values=np.array([1.0, 2.0, 3.0, q_max]),
                )
            )
        assert len(metrics._step_q_means) == 2
        assert metrics._step_q_maxes[1] == pytest.approx(8.0)

    def test_on_episode_end_stores_metrics(self):
        metrics = TrainingMetrics()

        # 스텝 데이터 시뮬레이션
        for q in [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]:
            metrics.on_step(
                StepInfo(
                    step_num=0,
                    state=np.zeros((4, 4)),
                    action=0,
                    reward=4.0,
                    loss=0.1,
                    q_values=np.array(q),
                )
            )

        result = EpisodeResult(
            episode_num=1,
            steps=10,
            score=256.0,
            max_tile=128,
            losses=[0.1, 0.2, 0.3],
            epsilon=0.95,
        )
        metrics.on_episode_end(result)

        assert metrics.episodes == [1]
        assert metrics.scores == [256.0]
        assert metrics.max_tiles == [128]
        assert metrics.steps_list == [10]
        assert metrics.epsilons == [0.95]
        assert metrics.avg_losses[0] == pytest.approx(0.2)
        assert metrics.total_steps == 10
        assert metrics.best_score == 256.0
        assert metrics.best_tile == 128

    def test_on_episode_end_resets_step_accumulators(self):
        metrics = TrainingMetrics()
        metrics.on_step(
            StepInfo(
                step_num=0,
                state=np.zeros((4, 4)),
                action=0,
                reward=4.0,
                loss=0.1,
                q_values=np.array([1.0, 2.0, 3.0, 4.0]),
            )
        )

        result = EpisodeResult(
            episode_num=1, steps=5, score=100.0, max_tile=64, losses=[0.1], epsilon=0.9
        )
        metrics.on_episode_end(result)

        assert len(metrics._step_q_means) == 0
        assert len(metrics._step_q_maxes) == 0

    def test_on_episode_end_empty_losses(self):
        metrics = TrainingMetrics()
        result = EpisodeResult(
            episode_num=1, steps=5, score=100.0, max_tile=64, losses=[], epsilon=0.9
        )
        metrics.on_episode_end(result)
        assert metrics.avg_losses[0] == 0.0

    def test_on_episode_end_no_q_values(self):
        """스텝 콜백 없이 에피소드 종료"""
        metrics = TrainingMetrics()
        result = EpisodeResult(
            episode_num=1, steps=3, score=50.0, max_tile=32, losses=[0.5], epsilon=0.8
        )
        metrics.on_episode_end(result)
        assert metrics.mean_q_values[0] == 0.0
        assert metrics.max_q_values[0] == 0.0

    def test_best_tracking(self):
        metrics = TrainingMetrics()
        for i, (score, tile) in enumerate([(100, 64), (500, 256), (200, 128)]):
            result = EpisodeResult(
                episode_num=i + 1,
                steps=10,
                score=score,
                max_tile=tile,
                losses=[0.1],
                epsilon=0.9,
            )
            metrics.on_episode_end(result)

        assert metrics.best_score == 500
        assert metrics.best_tile == 256

    def test_weight_norms_collected(self):
        """모델이 주어지면 가중치 노름 수집"""
        metrics = TrainingMetrics()

        class FakeModel:
            w1 = np.ones((16, 128))
            w2 = np.ones((128, 128))
            w3 = np.ones((128, 4))

        result = EpisodeResult(
            episode_num=1, steps=5, score=100.0, max_tile=64, losses=[0.1], epsilon=0.9
        )
        metrics.on_episode_end(result, model=FakeModel())

        assert len(metrics.weight_norms) == 1
        assert "w1" in metrics.weight_norms[0]
        assert "w2" in metrics.weight_norms[0]
        assert "w3" in metrics.weight_norms[0]
        assert metrics.weight_norms[0]["w1"] > 0

    def test_weight_norms_skipped_without_model(self):
        metrics = TrainingMetrics()
        result = EpisodeResult(
            episode_num=1, steps=5, score=100.0, max_tile=64, losses=[0.1], epsilon=0.9
        )
        metrics.on_episode_end(result, model=None)
        assert len(metrics.weight_norms) == 0

    def test_to_dict_structure(self):
        metrics = TrainingMetrics()
        result = EpisodeResult(
            episode_num=1,
            steps=10,
            score=200.0,
            max_tile=128,
            losses=[0.1, 0.2],
            epsilon=0.95,
        )
        metrics.on_episode_end(result)

        d = metrics.to_dict()

        # summary 키 검증
        assert "summary" in d
        summary = d["summary"]
        assert summary["total_episodes"] == 1
        assert summary["best_score"] == 200.0
        assert summary["best_tile"] == 128
        assert summary["current_epsilon"] == 0.95

        # 데이터 배열 키 검증
        for key in [
            "episodes",
            "scores",
            "score_ma",
            "max_tiles",
            "steps_list",
            "avg_losses",
            "loss_ma",
            "epsilons",
            "mean_q_values",
            "max_q_values",
            "weight_norms",
            "tile_distribution",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_tile_distribution(self):
        metrics = TrainingMetrics()
        tiles = [64, 128, 128, 256, 128]
        for i, tile in enumerate(tiles):
            result = EpisodeResult(
                episode_num=i + 1,
                steps=10,
                score=100.0,
                max_tile=tile,
                losses=[0.1],
                epsilon=0.9,
            )
            metrics.on_episode_end(result)

        d = metrics.to_dict()
        assert d["tile_distribution"] == {64: 1, 128: 3, 256: 1}

    def test_to_dict_empty(self):
        """데이터 없을 때 to_dict"""
        metrics = TrainingMetrics()
        d = metrics.to_dict()
        assert d["summary"]["total_episodes"] == 0
        assert d["summary"]["current_epsilon"] == 1.0
        assert d["episodes"] == []

    def test_to_dict_downsampling(self):
        """DOWNSAMPLE_THRESHOLD 초과 시 다운샘플링"""
        metrics = TrainingMetrics()
        for i in range(3000):
            result = EpisodeResult(
                episode_num=i + 1,
                steps=10,
                score=float(i),
                max_tile=64,
                losses=[0.1],
                epsilon=0.9,
            )
            metrics.on_episode_end(result)

        d = metrics.to_dict()
        # 다운샘플된 배열은 DOWNSAMPLE_THRESHOLD 이하
        assert len(d["episodes"]) <= 2000
        # 마지막 에피소드는 항상 포함
        assert d["episodes"][-1] == 3000
        # summary는 전체 데이터 기반
        assert d["summary"]["total_episodes"] == 3000


# --- 유틸 함수 테스트 ---


class TestMovingAverage:
    def test_empty(self):
        assert _moving_average([], 10) == []

    def test_single_value(self):
        assert _moving_average([5.0], 3) == [5.0]

    def test_window_larger_than_data(self):
        result = _moving_average([1.0, 2.0, 3.0], 10)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.5)
        assert result[2] == pytest.approx(2.0)

    def test_window_calculation(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = _moving_average(data, 3)
        assert result[0] == pytest.approx(10.0)  # [10]
        assert result[1] == pytest.approx(15.0)  # [10, 20]
        assert result[2] == pytest.approx(20.0)  # [10, 20, 30]
        assert result[3] == pytest.approx(30.0)  # [20, 30, 40]
        assert result[4] == pytest.approx(40.0)  # [30, 40, 50]


class TestDownsampleIndices:
    def test_no_downsample_needed(self):
        assert _downsample_indices(100, 200) == list(range(100))

    def test_exact_match(self):
        assert _downsample_indices(100, 100) == list(range(100))

    def test_downsample(self):
        indices = _downsample_indices(1000, 100)
        assert len(indices) == 100
        assert indices[0] == 0
        assert indices[-1] == 999  # 마지막 요소 포함
        # 단조 증가
        for i in range(1, len(indices)):
            assert indices[i] > indices[i - 1]


# --- TrainingVisualizer 테스트 ---


class TestTrainingVisualizer:
    def test_registers_callbacks(self):
        config = TrainConfig(method="td")
        trainer = create_trainer(config)
        assert trainer.on_step_callback is None
        assert trainer.on_episode_end_callback is None

        viz = TrainingVisualizer(trainer, port=9999)

        assert trainer.on_step_callback is not None
        assert trainer.on_episode_end_callback is not None

    def test_preserves_existing_callbacks(self):
        config = TrainConfig(method="td")
        trainer = create_trainer(config)

        original_called = {"step": False, "episode": False}

        def orig_step(info):
            original_called["step"] = True

        def orig_episode(result):
            original_called["episode"] = True

        trainer.on_step_callback = orig_step
        trainer.on_episode_end_callback = orig_episode

        viz = TrainingVisualizer(trainer, port=9998)

        # 콜백 실행 시 원래 콜백도 호출되는지 확인
        step_info = StepInfo(
            step_num=0,
            state=np.zeros((4, 4)),
            action=0,
            reward=0.0,
            loss=0.1,
            q_values=np.array([1.0, 2.0, 3.0, 4.0]),
        )
        trainer.on_step_callback(step_info)
        assert original_called["step"]

        episode_result = EpisodeResult(
            episode_num=1, steps=5, score=100.0, max_tile=64, losses=[0.1], epsilon=0.9
        )
        trainer.on_episode_end_callback(episode_result)
        assert original_called["episode"]

    def test_collects_metrics_during_training(self):
        config = TrainConfig(method="td", epsilon_start=1.0, epsilon_decay=0.99)
        trainer = create_trainer(config)
        viz = TrainingVisualizer(trainer, port=9997)

        # 실제 학습 실행 (짧게)
        trainer.train(episodes=5, print_every=5)

        assert len(viz.metrics.episodes) == 5
        assert len(viz.metrics.scores) == 5
        assert viz.metrics.total_steps > 0
        assert viz.metrics.best_tile >= 2

    def test_api_metrics_endpoint(self):
        config = TrainConfig(method="td")
        trainer = create_trainer(config)
        viz = TrainingVisualizer(trainer, port=9996)

        trainer.train(episodes=3, print_every=3)

        from fastapi.testclient import TestClient

        client = TestClient(viz.app)
        response = client.get("/api/metrics")
        assert response.status_code == 200

        data = response.json()
        assert data["summary"]["total_episodes"] == 3
        assert len(data["episodes"]) == 3

    def test_dashboard_html_endpoint(self):
        config = TrainConfig(method="td")
        trainer = create_trainer(config)
        viz = TrainingVisualizer(trainer, port=9995)

        from fastapi.testclient import TestClient

        client = TestClient(viz.app)
        response = client.get("/")
        assert response.status_code == 200
        assert "Training Dashboard" in response.text
        assert "Chart" in response.text

    def test_mc_trainer_integration(self):
        """MC 트레이너와 통합 테스트"""
        config = TrainConfig(method="mc")
        trainer = create_trainer(config)
        viz = TrainingVisualizer(trainer, port=9994)

        trainer.train(episodes=3, print_every=3)

        assert len(viz.metrics.episodes) == 3
        assert all(loss >= 0 for loss in viz.metrics.avg_losses)
