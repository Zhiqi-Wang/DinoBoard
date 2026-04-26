export function createAutoPlayController(stepFn, intervalMs = 700) {
  let timer = null;
  let running = false;
  let busy = false;

  async function tick() {
    if (!running || busy) return;
    busy = true;
    try {
      const keepRunning = await stepFn();
      if (keepRunning === false) {
        stop();
      }
    } finally {
      busy = false;
    }
  }

  function start() {
    if (running) return;
    running = true;
    timer = setInterval(tick, intervalMs);
    tick();
  }

  function stop() {
    running = false;
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  function isRunning() {
    return running;
  }

  return { start, stop, isRunning };
}
