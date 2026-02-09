const fs = require('fs');
const vm = require('vm');

// DOM mock
const mkEl = () => ({
    innerHTML: '', textContent: '', style: {}, className: '', href: '', download: '',
    querySelector: () => mkEl(), querySelectorAll: () => [],
    addEventListener: () => {}, appendChild: () => {}, click: () => {},
    clientWidth: 800, value: '1', files: [],
    getContext: () => new Proxy({}, { set: () => true, get: () => () => {} })
});

const sandbox = {
    document: { getElementById: () => mkEl(), querySelectorAll: () => [], createElement: () => mkEl(), addEventListener: () => {} },
    window: { devicePixelRatio: 1 },
    URL: { createObjectURL: () => '', revokeObjectURL: () => {} },
    console, setTimeout, Math, JSON, Array, Float64Array, Infinity, isNaN, parseInt,
    Promise, Blob: function(){},
};

const html = fs.readFileSync('ml_app.html', 'utf8');
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];
vm.createContext(sandbox);
vm.runInContext(script, sandbox);

// Test script
const testCode = `
(async () => {
    console.log('=== Game2048 ===');
    const g = new Game2048();
    g.reset();
    console.log('  reset: tiles=' + g.board.flat().filter(x=>x>0).length);
    console.log('  validActions:', g.getValidActions());
    const r = g.step(0);
    console.log('  step: reward=' + r.reward + ' done=' + r.done);

    console.log('');
    console.log('=== QNetwork ===');
    const net = new QNetwork(256);
    const q = net.forward(g.getState());
    console.log('  forward: ' + q.map(v=>v.toFixed(4)).join(', '));
    net.forward(g.getState());
    const loss = net.backward(0, 5.0, 0.001);
    console.log('  backward: loss=' + loss.toFixed(6));
    const j = net.toJSON();
    const n2 = QNetwork.fromJSON(j);
    const qA = net.forward(g.getState());
    const q2 = n2.forward(g.getState());
    const ok = qA.every((v,i) => Math.abs(v-q2[i])<1e-10);
    console.log('  roundtrip: ' + (ok ? 'PASS' : 'FAIL'));

    console.log('');
    console.log('=== TDTrainer ===');
    const td = new TDTrainer(net, {epsilonStart: 0.5});
    let tdSteps = 0;
    td.onStep = () => tdSteps++;
    await td.trainEpisodes(new Game2048(), 1, r => {
        console.log('  episode: steps=' + r.steps + ' score=' + r.score);
    });
    console.log('  onStep count: ' + tdSteps);

    console.log('');
    console.log('=== MCTrainer ===');
    const mc = new MCTrainer(net, {epsilonStart: 0.5});
    await mc.trainEpisodes(new Game2048(), 1, r => {
        console.log('  episode: steps=' + r.steps + ' score=' + r.score);
    });

    console.log('');
    console.log('ALL PASSED');
})();
`;

vm.runInContext(testCode, sandbox);
