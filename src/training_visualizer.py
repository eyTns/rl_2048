"""머신러닝 학습 과정 시각화 모듈

학습 중 메트릭을 수집하고 실시간 웹 대시보드로 시각화합니다.

사용법:
    trainer = create_trainer(config)
    viz = TrainingVisualizer(trainer, port=8888)
    viz.start()  # 대시보드 서버 시작
    trainer.train(episodes=1000)
"""

import json
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .trainer import BaseTrainer, EpisodeResult, StepInfo

MOVING_AVERAGE_WINDOW = 50
DOWNSAMPLE_THRESHOLD = 2000


class TrainingMetrics:
    """학습 메트릭 수집기

    트레이너 콜백을 통해 에피소드별/스텝별 메트릭을 수집하고
    시각화에 필요한 집계 데이터를 제공합니다.
    """

    def __init__(self):
        # 에피소드별 메트릭
        self.episodes: list[int] = []
        self.scores: list[float] = []
        self.max_tiles: list[int] = []
        self.steps_list: list[int] = []
        self.avg_losses: list[float] = []
        self.epsilons: list[float] = []

        # Q값 추적 (에피소드별 집계)
        self.mean_q_values: list[float] = []
        self.max_q_values: list[float] = []

        # 가중치 노름 추적
        self.weight_norms: list[dict[str, float]] = []

        # 에피소드 내 스텝 누적기 (에피소드마다 리셋)
        self._step_q_means: list[float] = []
        self._step_q_maxes: list[float] = []

        # 전체 통계
        self.total_steps = 0
        self.best_score = 0.0
        self.best_tile = 0

    def on_step(self, info: StepInfo):
        """스텝 콜백: Q값 수집"""
        if info.q_values is not None:
            q = info.q_values
            self._step_q_means.append(float(np.mean(q)))
            self._step_q_maxes.append(float(np.max(q)))

    def on_episode_end(self, result: EpisodeResult, model=None):
        """에피소드 종료 콜백: 에피소드 메트릭 저장"""
        self.episodes.append(result.episode_num)
        self.scores.append(result.score)
        self.max_tiles.append(result.max_tile)
        self.steps_list.append(result.steps)
        self.epsilons.append(result.epsilon)

        avg_loss = float(np.mean(result.losses)) if result.losses else 0.0
        self.avg_losses.append(avg_loss)

        mean_q = (
            float(np.mean(self._step_q_means)) if self._step_q_means else 0.0
        )
        max_q = float(np.max(self._step_q_maxes)) if self._step_q_maxes else 0.0
        self.mean_q_values.append(mean_q)
        self.max_q_values.append(max_q)

        if model:
            self.weight_norms.append(
                {
                    "w1": float(np.linalg.norm(model.w1)),
                    "w2": float(np.linalg.norm(model.w2)),
                    "w3": float(np.linalg.norm(model.w3)),
                }
            )

        self.total_steps += result.steps
        self.best_score = max(self.best_score, result.score)
        self.best_tile = max(self.best_tile, result.max_tile)

        self._step_q_means.clear()
        self._step_q_maxes.clear()

    def to_dict(self) -> dict:
        """JSON 직렬화 가능한 딕셔너리 반환"""
        score_ma = _moving_average(self.scores, MOVING_AVERAGE_WINDOW)
        loss_ma = _moving_average(self.avg_losses, MOVING_AVERAGE_WINDOW)

        tile_counts: dict[int, int] = defaultdict(int)
        for tile in self.max_tiles:
            tile_counts[tile] += 1
        tile_dist = dict(sorted(tile_counts.items()))

        episodes = self.episodes
        scores = self.scores
        max_tiles = self.max_tiles
        steps_list = self.steps_list
        avg_losses = self.avg_losses
        epsilons = self.epsilons
        mean_q = self.mean_q_values
        max_q = self.max_q_values
        w_norms = self.weight_norms

        # 데이터가 많으면 다운샘플링
        n = len(episodes)
        if n > DOWNSAMPLE_THRESHOLD:
            indices = _downsample_indices(n, DOWNSAMPLE_THRESHOLD)
            episodes = [episodes[i] for i in indices]
            scores = [scores[i] for i in indices]
            score_ma = [score_ma[i] for i in indices]
            max_tiles = [max_tiles[i] for i in indices]
            steps_list = [steps_list[i] for i in indices]
            avg_losses = [avg_losses[i] for i in indices]
            loss_ma = [loss_ma[i] for i in indices]
            epsilons = [epsilons[i] for i in indices]
            mean_q = [mean_q[i] for i in indices]
            max_q = [max_q[i] for i in indices]
            w_norms = [w_norms[i] for i in indices] if w_norms else []

        return {
            "summary": {
                "total_episodes": len(self.episodes),
                "total_steps": self.total_steps,
                "best_score": self.best_score,
                "best_tile": self.best_tile,
                "current_epsilon": self.epsilons[-1] if self.epsilons else 1.0,
                "latest_avg_score": (
                    float(np.mean(self.scores[-MOVING_AVERAGE_WINDOW :]))
                    if self.scores
                    else 0
                ),
            },
            "episodes": episodes,
            "scores": scores,
            "score_ma": score_ma,
            "max_tiles": max_tiles,
            "steps_list": steps_list,
            "avg_losses": avg_losses,
            "loss_ma": loss_ma,
            "epsilons": epsilons,
            "mean_q_values": mean_q,
            "max_q_values": max_q,
            "weight_norms": w_norms,
            "tile_distribution": tile_dist,
        }


def _moving_average(data: list[float], window: int) -> list[float]:
    """이동 평균 계산"""
    if not data:
        return []
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(data[start : i + 1])))
    return result


def _downsample_indices(n: int, target: int) -> list[int]:
    """균등 간격으로 다운샘플링할 인덱스 반환 (마지막 요소 항상 포함)"""
    if n <= target:
        return list(range(n))
    step = n / target
    indices = [int(i * step) for i in range(target - 1)]
    indices.append(n - 1)
    return indices


class TrainingVisualizer:
    """학습 과정 실시간 웹 대시보드

    트레이너에 콜백을 등록하여 메트릭을 수집하고,
    FastAPI 기반 웹 대시보드에서 실시간으로 시각화합니다.

    사용법:
        viz = TrainingVisualizer(trainer, port=8888)
        viz.start()
        trainer.train(episodes=1000)
    """

    def __init__(self, trainer: BaseTrainer, port: int = 8888):
        self.trainer = trainer
        self.port = port
        self.metrics = TrainingMetrics()
        self._server_thread: threading.Thread | None = None

        self._register_callbacks(trainer)
        self.app = self._create_app()

    def _register_callbacks(self, trainer: BaseTrainer):
        """트레이너에 메트릭 수집 콜백 등록 (기존 콜백 보존)"""
        original_step_cb = trainer.on_step_callback
        original_episode_cb = trainer.on_episode_end_callback

        def step_callback(info: StepInfo):
            self.metrics.on_step(info)
            if original_step_cb:
                original_step_cb(info)

        def episode_callback(result: EpisodeResult):
            self.metrics.on_episode_end(result, model=trainer.model)
            if original_episode_cb:
                original_episode_cb(result)

        trainer.on_step_callback = step_callback
        trainer.on_episode_end_callback = episode_callback

    def _create_app(self) -> FastAPI:
        """FastAPI 앱 생성"""
        app = FastAPI(title="2048 RL Training Dashboard")

        @app.get("/", response_class=HTMLResponse)
        def dashboard():
            html_path = Path(__file__).parent.parent / "static" / "training_dashboard.html"
            return html_path.read_text(encoding="utf-8")

        @app.get("/api/metrics")
        def get_metrics():
            return self.metrics.to_dict()

        return app

    def start(self):
        """대시보드 서버를 백그라운드 스레드로 시작"""
        self._server_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": self.app,
                "host": "0.0.0.0",
                "port": self.port,
                "log_level": "warning",
            },
            daemon=True,
        )
        self._server_thread.start()
        print(f"Training dashboard: http://localhost:{self.port}")

    def export_html(self, path: str = "training_report.html"):
        """학습 결과를 정적 HTML 파일로 내보내기

        서버 없이 브라우저에서 바로 열 수 있는 독립 파일을 생성합니다.
        모바일 브라우저에서도 열 수 있고, GitHub Pages 등에 배포할 수 있습니다.

        Args:
            path: 저장할 파일 경로
        """
        template_path = Path(__file__).parent.parent / "static" / "training_dashboard.html"
        html = template_path.read_text(encoding="utf-8")

        data_json = json.dumps(self.metrics.to_dict(), ensure_ascii=False)
        data_script = f"<script>window.__STATIC_DATA__ = {data_json};</script>"

        # </head> 바로 앞에 데이터 스크립트 삽입
        html = html.replace("</head>", f"{data_script}\n</head>")

        Path(path).write_text(html, encoding="utf-8")
        print(f"Training report exported: {path}")


if __name__ == "__main__":
    from .trainer import TrainConfig, create_trainer

    print("=== 학습 시각화 데모 ===")
    config = TrainConfig(method="td", gamma=0.999999)
    trainer = create_trainer(config)

    viz = TrainingVisualizer(trainer, port=8888)
    viz.start()

    trainer.train(episodes=500, print_every=50)

    # 정적 HTML 내보내기 (모바일에서 열 수 있음)
    viz.export_html("training_report.html")
    print("학습 완료. training_report.html을 모바일로 공유하세요.")

    # 서버 유지
    try:
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        print("종료")
