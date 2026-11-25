"""
Internal logging utility.
"""

import logging
import os
import requests
import json
from datetime import datetime, timezone
import re
from collections.abc import Mapping, Sequence, Set

from loguru import logger
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from common.utils import unwrap

_w = os.getenv("TABBY_LOG_CONSOLE_WIDTH")
_default_console_width = int(_w) if _w is not None and _w.isnumeric() else None
RICH_CONSOLE = Console(width=_default_console_width)
LOG_LEVEL = os.getenv("TABBY_LOG_LEVEL", "INFO")


def get_progress_bar():
    return Progress(console=RICH_CONSOLE)


def get_loading_progress_bar():
    """Gets a pre-made progress bar for loading tasks."""

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=RICH_CONSOLE,
    )


def _log_formatter(record: dict):
    """Log message formatter."""

    color_map = {
        "TRACE": "dim blue",
        "DEBUG": "cyan",
        "INFO": "green",
        "SUCCESS": "bold green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold white on red",
    }

    time = record.get("time")
    colored_time = f"[grey37]{time:YYYY-MM-DD HH:mm:ss.SSS}[/grey37]"

    level = record.get("level")
    level_color = color_map.get(level.name, "cyan")
    colored_level = f"[{level_color}]{level.name}[/{level_color}]:"

    separator = " " * (9 - len(level.name))

    message = unwrap(record.get("message"), "")

    # Replace once loguru allows for turning off str.format
    message = message.replace("{", "{{").replace("}", "}}").replace("<", r"\<")

    # Escape markup tags from Rich
    message = escape(message)
    lines = message.splitlines()

    fmt = ""
    if len(lines) > 1:
        fmt = "\n".join([f"{colored_time} {colored_level}{separator}{line}" for line in lines])
    else:
        fmt = f"{colored_time} {colored_level}{separator}{message}"

    return fmt


# Uvicorn log handler
# Uvicorn log portions inspired from https://github.com/encode/uvicorn/discussions/2027#discussioncomment-6432362
class UvicornLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger.opt(exception=record.exc_info).log(record.levelname, self.format(record).rstrip())


# Uvicorn config for logging. Passed into run when creating all loggers in server
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "uvicorn": {
            "class": f"{UvicornLoggingHandler.__module__}.{UvicornLoggingHandler.__qualname__}",  # noqa
        },
    },
    "root": {"handlers": ["uvicorn"], "propagate": False, "level": LOG_LEVEL},
}


def setup_logger():
    """Bootstrap the logger."""

    logger.remove()

    logger.add(
        RICH_CONSOLE.print,
        level=LOG_LEVEL,
        format=_log_formatter,
        colorize=True,
    )
    # Add file logging
    logger.add(
        "logs/{time}.log",
        level=LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="20 MB",  # Rotate file when it reaches 20MB
        retention="1 week",  # Keep logs for 1 week
        compression="zip",  # Compress rotated log
    )


def setup_console_tee(path: str):
    """Mirror stdout + stderr to a file at the DESCRIPTOR level, keeping the terminal.

    The loguru file sink only sees loguru records. A TP worker that dies writes its
    traceback with a raw sys.stderr.write in its OWN process ([rankN] worker error),
    and native CUDA/NCCL/segfault messages never pass through Python logging at all.
    Spawned workers inherit fds 1/2, so tee'ing the descriptors captures them too.
    Opt-in via logging.log_file; POSIX only (select on pipes).
    """

    import sys
    import threading

    if os.name != "posix":
        logger.warning(f"log_file console tee is POSIX-only; {path} not enabled")
        return

    try:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        logf = open(path, "ab", buffering=0)

        import select

        r1, w1 = os.pipe()  # stdout
        r2, w2 = os.pipe()  # stderr
        out1, out2 = os.dup(1), os.dup(2)
        os.dup2(w1, 1)
        os.dup2(w2, 2)
        os.close(w1)
        os.close(w2)

        # Spawned workers are fresh interpreters inheriting the tee'd fds; make
        # them unbuffered so their tracebacks hit the file immediately.
        os.environ["PYTHONUNBUFFERED"] = "1"
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass
        # rich decided isatty at first use; the pipe is not a tty, so keep colors.
        RICH_CONSOLE._force_terminal = True

        stop = threading.Event()

        def _tee():
            routes = {r1: out1, r2: out2}
            term_dead = set()
            while routes:
                try:
                    ready, _, _ = select.select(list(routes), [], [], 0.2)
                except OSError:
                    break
                if not ready:
                    if stop.is_set():
                        break
                    continue
                for rr in ready:
                    try:
                        data = os.read(rr, 65536)
                    except OSError:
                        data = b""
                    if not data:
                        os.close(rr)
                        del routes[rr]
                        continue
                    try:
                        logf.write(data)
                    except Exception:
                        pass
                    fd = routes[rr]
                    if fd not in term_dead:
                        try:
                            os.write(fd, data)
                        except Exception:
                            term_dead.add(fd)

        tee_thread = threading.Thread(target=_tee, daemon=True, name="console-tee")
        tee_thread.start()

        # Drain pending output at exit so the last lines before shutdown are not lost.
        import atexit

        def _drain():
            stop.set()
            tee_thread.join(timeout=2.0)

        atexit.register(_drain)
        logger.info(f"Console output (stdout+stderr, incl. TP workers) mirrored to {path}")
    except Exception as e:
        logger.warning(f"log_file console tee failed ({e}); using the loguru logs/ sink only")


"""
Extended logging via Seq.
"""

_DATA_URL_RE = re.compile(r"^(data:)([^;,]+)?(?:;[^,]*)?(;base64),(.*)$", re.DOTALL)


def _sanitize_for_logging(obj, head=1024, tail=1024):
    def truncate_string(s: str) -> str:
        if head + tail >= len(s):
            return s

        omitted = len(s) - head - tail
        return f"{s[:head]} [<- {omitted:,} chars truncated ->] {s[-tail:]}"

    def sanitize_string(s: str) -> str:
        m = _DATA_URL_RE.match(s)
        if m:
            prefix1, mime_type, prefix3, payload = m.groups()
            mime_type = mime_type or "application/octet-stream"
            prefix = f"{prefix1}{mime_type}{prefix3}"
            return f"{prefix} [<- {len(payload):,} chars truncated ->]"

        return truncate_string(s)

    def walk(value):
        if isinstance(value, str):
            return sanitize_string(value)

        if isinstance(value, Mapping):
            return {k: walk(v) for k, v in value.items()}

        if isinstance(value, tuple):
            return tuple(walk(v) for v in value)

        if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
            return {walk(v) for v in value}

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [walk(v) for v in value]

        return value

    return walk(obj)


class XLogger:
    def __init__(self):
        self.seqlog_url = None
        self.headers = {}
        self.enabled = False

    def _get_timestamp_now(self):
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def setup(self, seqlog_url: str = "http://localhost:5341", api_key: str | None = None):
        self.seqlog_url = seqlog_url.rstrip("/")
        self.headers = {"Content-Type": "application/vnd.serilog.clef"}
        if api_key:
            self.headers["X-Seq-ApiKey"] = api_key

        # Check if seqlog is reachable
        try:
            r = requests.post(
                self.seqlog_url + "/ingest/clef",
                data=(f'{{"@t":"{self._get_timestamp_now()}","@m":"TabbyAPI startup probe"}}\n'),
                headers=self.headers,
                timeout=2,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            logger.info(
                f"Failed to initialize seqlog handler for server at "
                f"{self.seqlog_url}: {e}"
                f"seqlog logging is disabled."
            )
            return

        self.enabled = True
        logger.info(f"Enabled logging to seqlog instance at {self.seqlog_url}")

    def _commit(self, log_level: str, log_message: str, log_extra: dict):
        if not self.enabled:
            return

        try:
            if log_extra is None:
                log_extra = {}
            elif not isinstance(log_extra, dict):
                log_extra = {"extra": str(log_extra)}
            log_extra = _sanitize_for_logging(log_extra)
            event = {
                "@t": self._get_timestamp_now(),
                "@m": log_message,
                "@l": log_level,
                **log_extra,
            }
            try:
                data = json.dumps(event, default=str) + "\n"
            except Exception as e:
                data = "## Failed to serialize log data: " + str(e)
            r = requests.post(
                self.seqlog_url + "/ingest/clef",
                data=data,
                headers=self.headers,
                timeout=2,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to write log event to Seq, logging disabled: {e}")
            self.enabled = False

    def _compose(self, log_message, details):
        return (log_message + " " + details) if details else log_message

    def verbose(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        self._commit("Verbose", log_message, log_extra)

    def debug(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.debug(self._compose(log_message, details))
        self._commit("Debug", log_message, log_extra)

    def info(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.info(self._compose(log_message, details))
        self._commit("Information", log_message, log_extra)

    def warning(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.warning(self._compose(log_message, details))
        self._commit("Warning", log_message, log_extra)

    def error(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.error(self._compose(log_message, details))
        self._commit("Error", log_message, log_extra)


xlogger = XLogger()
