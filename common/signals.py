import asyncio
import os
import signal
import threading
import time
from common.logger import xlogger
from types import FrameType

SHUTTING_DOWN: bool = False


def signal_handler(*_):
    """SIGINT/SIGTERM: unload the model, then GUARANTEE process exit.

    The old handler called loop.run_until_complete() on the already-RUNNING uvicorn
    loop (RuntimeError) and fell back to sys.exit(0) from inside the signal handler.
    Interpreter exit then joined the non-daemon job-thread executors, which were
    parked in worker pipes / NCCL collectives -- a permanent freeze with the model's
    VRAM still held. Now:

      * schedule the async unload on the running loop (never run_until_complete),
      * arm a watchdog that hard-exits after 15 s no matter what hangs,
      * a second signal hard-exits immediately.

    Workers are quit/killed by TPProcessGroup.close(); any straggler self-exits via
    its parent-PID watchdog once this process dies, so VRAM is always released.
    """

    global SHUTTING_DOWN

    if SHUTTING_DOWN:
        os._exit(1)
    SHUTTING_DOWN = True

    xlogger.warning("Shutdown signal called. Exiting gracefully.")

    def _watchdog():
        time.sleep(15)
        os._exit(0)
    threading.Thread(target = _watchdog, daemon = True).start()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(signal_handler_async()))
    else:
        try:
            asyncio.run(signal_handler_async())
        except Exception:
            pass
        os._exit(0)


async def signal_handler_async():
    from common import model

    """
    Internal signal handler. Runs all async code to shut down the program.
    Ends in a hard exit so interpreter shutdown can never hang on executor threads.
    """

    try:
        if model.container:
            await model.unload_model(skip_wait = True, shutdown = True)

        if model.embeddings_container:
            await model.unload_embedding_model()
    except Exception as e:
        xlogger.warning(f"Shutdown unload error: {e}")

    # Hard exit: bypasses interpreter-shutdown joins of executor threads that may
    # still be parked in worker pipes / NCCL collectives. All GPU resources are
    # released by the kernel on process exit.
    os._exit(0)


def uvicorn_signal_handler(signal_event: signal.Signals):
    """Overrides uvicorn's signal handler."""

    default_signal_handler = signal.getsignal(signal_event)

    def wrapped_handler(signum: int, frame: FrameType = None):
        xlogger.warning("Shutdown signal called. Exiting gracefully.")
        default_signal_handler(signum, frame)

    signal.signal(signal_event, wrapped_handler)