// Runs bot.js against a local replica of play2048.co/classic: same rules, same
// encoded localStorage save, same keydown handling. No browser needed.
// Needs `g2048 serve` running. Usage: node bot/replica-test.mjs [games=3]
import { readFileSync } from 'node:fs';

const KEY = new TextEncoder().encode('dGhlIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw==');
// Inverse of the bot's decoder, mirroring the site's Ka(): XOR, base64, padding shifted.
const encode = obj => {
  const t = new TextEncoder().encode(JSON.stringify(obj));
  const x = Uint8Array.from(t, (c, i) => c ^ KEY[i % KEY.length]);
  const s = Buffer.from(x).toString('base64');
  const n = s.endsWith('==') ? 2 : s.endsWith('=') ? 1 : 0;
  return s.substring(0, s.length - n) + ['', '=', '=='][(n + 2) % 3];
};

class Game {
  constructor() {
    this.board = Array.from({ length: 4 }, () => Array(4).fill(0));
    this.score = 0; this.moveCount = 0; this.state = 'fresh'; this.won = false;
    this.spawn(); this.spawn();
  }
  spawn() {
    const empty = [];
    this.board.forEach((r, y) => r.forEach((v, x) => v || empty.push([y, x])));
    const [y, x] = empty[Math.floor(Math.random() * empty.length)];
    this.board[y][x] = Math.random() < 0.9 ? 2 : 4;
  }
  // Slide toward index 0; a merged tile can't merge again this move.
  slide(line) {
    const v = line.filter(Boolean), out = [];
    for (let i = 0; i < v.length; i++) {
      if (v[i] === v[i + 1]) { out.push(v[i] * 2); this.score += v[i] * 2; i++; } else out.push(v[i]);
    }
    return [...out, 0, 0, 0, 0].slice(0, 4);
  }
  move(dir) {
    if (this.state === 'gameOver' || this.state === 'gameWon') return;
    const b = this.board, before = JSON.stringify(b);
    for (let i = 0; i < 4; i++) {
      const idx = [0, 1, 2, 3].map(j => ({
        left: [i, j], right: [i, 3 - j], up: [j, i], down: [3 - j, i],
      }[dir]));
      const res = this.slide(idx.map(([y, x]) => b[y][x]));
      idx.forEach(([y, x], k) => (b[y][x] = res[k]));
    }
    if (JSON.stringify(b) === before) return;
    this.moveCount++;
    this.spawn();
    if (!this.won && b.flat().includes(2048)) { this.won = true; this.state = 'gameWon'; }
    else this.state = this.canMove() ? 'playing' : 'gameOver';
  }
  canMove() {
    const b = this.board;
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
      if (!b[y][x] || b[y][x] === b[y][x + 1] || (y < 3 && b[y][x] === b[y + 1][x])) return true;
    }
    return false;
  }
  // Same shape the site saves: board[row][col] = {value} | null, and no power-ups in classic.
  save() {
    const board = this.board.map(r => r.map(v => (v ? { value: v, id: 'x' } : null)));
    store.set('z291replicaclassic', encode({ state: this.state, board, moveCount: this.moveCount, score: this.score, powerups: {} }));
  }
}

const store = new Map();
store.set('unrelated-ad-key', 'not base64 at all');
globalThis.localStorage = {
  getItem: k => store.get(k) ?? null,
  setItem: (k, v) => store.set(k, String(v)),
  key: i => [...store.keys()][i],
  get length() { return store.size; },
};
// Object.keys(localStorage) must list the stored keys, like a real Storage object.
globalThis.localStorage = new Proxy(globalThis.localStorage, {
  ownKeys: () => [...store.keys()],
  getOwnPropertyDescriptor: () => ({ enumerable: true, configurable: true }),
});

let game;
const keyDir = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' };
const keepBtn = { textContent: 'Keep going', click() { game.state = 'playing'; game.save(); } };
globalThis.document = {
  dispatchEvent(e) {
    const d = keyDir[e.key];
    // Site saves asynchronously after the move animation starts.
    if (d) setTimeout(() => { game.move(d); game.save(); }, 5);
    return true;
  },
  querySelectorAll: () => (game.state === 'gameWon' ? [keepBtn] : []),
};
globalThis.KeyboardEvent = class { constructor(type, init) { Object.assign(this, { type }, init); } };
globalThis.window = globalThis;

const src = readFileSync(new URL('./bot.js', import.meta.url), 'utf8').replace('DELAY_MS = 60', 'DELAY_MS = 0')
  .replace('await sleep(20)', 'await sleep(1)')
  .replace('127.0.0.1:20480', `127.0.0.1:${process.env.PORT ?? 20480}`);
const logs = [];
if (process.env.DEBUG) setInterval(() => process.stderr.write(`moves ${game.moveCount} state ${game.state} last: ${logs.at(-1)}\n`), 2000);
const realLog = console.log;
console.log = (...a) => logs.push(a.join(' '));

const n = Number(process.argv[2] ?? 3);
let failures = 0;
for (let i = 0; i < n; i++) {
  game = new Game();
  game.save();
  logs.length = 0;
  const t0 = Date.now();
  eval(src);
  while (!logs.some(l => /game over|no move|stopping|not found|no classic/.test(l))) await new Promise(r => setTimeout(r, 50));
  const end = logs.at(-1);
  const max = Math.max(...game.board.flat());
  const ok = game.state === 'gameOver' && /game over/.test(end);
  if (!ok) failures++;
  realLog(`game ${i}: ${ok ? 'ok ' : 'FAIL'} score ${game.score} max ${max} moves ${game.moveCount} in ${((Date.now() - t0) / 1000).toFixed(0)}s  (${end})`);
}
realLog(failures ? `${failures} of ${n} games FAILED` : `all ${n} games ran to game over`);
process.exit(failures ? 1 : 0);
