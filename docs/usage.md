# 사용법

## 설치

```bash
# 기본 설치 (RL 학습용)
uv sync

# UI 포함 설치
uv sync  # pygame, fastapi, uvicorn 포함
```

## 실행

### 컴퓨터 UI (pygame)

```bash
uv run python game_ui.py
```

- 조작: 화살표 키 또는 WASD
- 재시작: R
- 종료: ESC

### 모바일/웹 UI (FastAPI 서버)

```bash
uv run python web_ui.py
```

- 접속: http://localhost:8000
- 조작: 터치 스와이프 또는 키보드
- 단일 유저 전용 (ML 연동용)

### 정적 웹 UI (서버 불필요)

브라우저에서 `static_ui.html` 파일 직접 열기

- GitHub Pages 배포 가능
- 조작: 터치 스와이프 또는 키보드

## API 엔드포인트 (web_ui.py)

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 게임 UI 페이지 |
| GET | `/state` | 현재 게임 상태 조회 |
| POST | `/move` | 이동 (body: `{"action": 0-3}`) |
| POST | `/reset` | 게임 리셋 |

action 값:
- 0: 위
- 1: 아래
- 2: 왼쪽
- 3: 오른쪽
