// 2048-solver browser bot for play2048.co/classic.
// 1. Run `g2048 serve --net nets/main.bin --depth 3` locally.
// 2. Open https://play2048.co/classic, paste this into the DevTools console.
// Stop with `g2048bot.stop()`. It only reads the saved game and presses arrow keys.
(() => {
  const SERVER = 'http://127.0.0.1:20480';
  const DELAY_MS = 60; // pause after each move so the site can save its state

  // The site stores each mode's game as base64 (padding shifted by one) XORed with this key, then JSON.
  const KEY = new TextEncoder().encode('dGhlIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw==');
  const fixPad = s => {
    const t = s.length;
    const n = t >= 2 && s.charCodeAt(t - 2) === 61 ? 2 : t >= 1 && s.charCodeAt(t - 1) === 61 ? 1 : 0;
    return s.substring(0, t - n) + ['', '=', '=='][(n + 1) % 3];
  };
  const decode = s => {
    const b = atob(fixPad(s));
    const u = Uint8Array.from(b, (c, i) => c.charCodeAt(0) ^ KEY[i % KEY.length]);
    return JSON.parse(new TextDecoder().decode(u));
  };

  // Classic games carry no power-ups; standard games do.
  const readGame = () => {
    for (const k of Object.keys(localStorage)) {
      try {
        const v = decode(localStorage.getItem(k));
        if (v && v.board && (!v.powerups || Object.keys(v.powerups).length === 0)) return v;
      } catch {}
    }
    return null;
  };

  const rank = c => (c && c.value ? Math.log2(c.value) : 0);
  const toHex = board => board.flat().map(c => Math.min(rank(c), 15).toString(16)).join('');

  const KEYS = { up: ['ArrowUp', 38], down: ['ArrowDown', 40], left: ['ArrowLeft', 37], right: ['ArrowRight', 39] };
  const press = dir => {
    const [key, code] = KEYS[dir];
    document.dispatchEvent(new KeyboardEvent('keydown', { key, code: key, keyCode: code, which: code, bubbles: true }));
  };

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  let running = true;

  (async () => {
    let lastMoves = -1, stuck = 0;
    while (running) {
      const g = readGame();
      if (!g) { console.log('[2048-solver] no classic game found'); break; }
      if (g.state === 'gameOver') { console.log(`[2048-solver] game over, score ${g.score}`); break; }
      if (g.state === 'gameWon') {
        // Reaching 2048 pauses the game behind a "keep playing" prompt.
        const btn = [...document.querySelectorAll('button, a')].find(b => /keep (going|playing)|continue/i.test(b.textContent));
        if (!btn) { console.log('[2048-solver] won, but no keep-playing button found'); break; }
        btn.click();
        await sleep(500);
        continue;
      }
      // Wait until the previous move has been saved before deciding the next one.
      if (g.moveCount === lastMoves) {
        if (++stuck > 100) { console.log('[2048-solver] board stopped changing, stopping'); break; }
        await sleep(20);
        continue;
      }
      stuck = 0;
      lastMoves = g.moveCount;
      const dir = await (await fetch(`${SERVER}/move?b=${toHex(g.board)}`)).text();
      if (dir === 'none' || !KEYS[dir]) { console.log(`[2048-solver] no move (${dir}), score ${g.score}`); break; }
      press(dir);
      await sleep(DELAY_MS);
      if (g.moveCount % 100 === 0) console.log(`[2048-solver] move ${g.moveCount}, score ${g.score}`);
    }
    running = false;
  })();

  window.g2048bot = { stop: () => { running = false; }, readGame };
  console.log('[2048-solver] running; g2048bot.stop() to stop');
})();
