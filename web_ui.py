from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from game2048 import Game2048

app = FastAPI()
game = Game2048()

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>2048</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #faf8ef; display: flex; justify-content: center; align-items: center; min-height: 100vh; touch-action: none; }
        .container { text-align: center; }
        h1 { color: #776e65; font-size: 48px; margin-bottom: 10px; }
        .score { color: #776e65; font-size: 24px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(4, 80px); gap: 10px; background: #bbada0; padding: 10px; border-radius: 6px; }
        .tile { width: 80px; height: 80px; display: flex; justify-content: center; align-items: center; font-size: 28px; font-weight: bold; border-radius: 4px; }
        .t0 { background: #cdc1b4; }
        .t2 { background: #eee4da; color: #776e65; }
        .t4 { background: #ede0c8; color: #776e65; }
        .t8 { background: #f2b179; color: #f9f6f2; }
        .t16 { background: #f59563; color: #f9f6f2; }
        .t32 { background: #f67c5f; color: #f9f6f2; }
        .t64 { background: #f65e3b; color: #f9f6f2; }
        .t128 { background: #edcf72; color: #f9f6f2; }
        .t256 { background: #edcc61; color: #f9f6f2; }
        .t512 { background: #edc850; color: #f9f6f2; }
        .t1024 { background: #edc53f; color: #f9f6f2; font-size: 22px; }
        .t2048 { background: #edc22e; color: #f9f6f2; font-size: 22px; }
        .tbig { background: #3c3a32; color: #f9f6f2; font-size: 18px; }
        .game-over { color: #776e65; font-size: 32px; margin-top: 20px; }
        button { margin-top: 20px; padding: 15px 30px; font-size: 18px; cursor: pointer; background: #8f7a66; color: white; border: none; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>2048</h1>
        <div class="score">Score: <span id="score">0</span></div>
        <div class="grid" id="grid"></div>
        <div class="game-over" id="gameover" style="display:none;">Game Over!</div>
        <button onclick="reset()">New Game</button>
    </div>
    <script>
        let touchStartX, touchStartY;
        async function move(action) {
            const res = await fetch('/move', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action}) });
            render(await res.json());
        }
        async function reset() {
            const res = await fetch('/reset', { method: 'POST' });
            render(await res.json());
        }
        async function load() {
            const res = await fetch('/state');
            render(await res.json());
        }
        function render(data) {
            document.getElementById('score').textContent = data.score;
            document.getElementById('gameover').style.display = data.done ? 'block' : 'none';
            const grid = document.getElementById('grid');
            grid.innerHTML = '';
            data.board.flat().forEach(v => {
                const tile = document.createElement('div');
                tile.className = 'tile ' + (v > 2048 ? 'tbig' : 't' + v);
                tile.textContent = v || '';
                grid.appendChild(tile);
            });
        }
        document.addEventListener('keydown', e => {
            const map = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3, w: 0, s: 1, a: 2, d: 3 };
            if (map[e.key] !== undefined) { e.preventDefault(); move(map[e.key]); }
        });
        document.addEventListener('touchstart', e => { touchStartX = e.touches[0].clientX; touchStartY = e.touches[0].clientY; });
        document.addEventListener('touchend', e => {
            const dx = e.changedTouches[0].clientX - touchStartX;
            const dy = e.changedTouches[0].clientY - touchStartY;
            if (Math.abs(dx) < 30 && Math.abs(dy) < 30) return;
            if (Math.abs(dx) > Math.abs(dy)) move(dx > 0 ? 3 : 2);
            else move(dy > 0 ? 1 : 0);
        });
        load();
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/state")
def state():
    return {"board": game.get_state().tolist(), "score": game.score, "done": game.done}


@app.post("/move")
def move(data: dict):
    game.step(data["action"])
    return {"board": game.get_state().tolist(), "score": game.score, "done": game.done}


@app.post("/reset")
def reset():
    game.reset()
    return {"board": game.get_state().tolist(), "score": game.score, "done": game.done}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
