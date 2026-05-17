"""Cross-platform file locking for shared resource access.

Provides advisory file locking using portalocker on Unix/Windows,
with a fallback to simple atomic operations when portalocker is
unavailable. Used for coordinating BM25 index access across
multiple processes.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

logger = get_logger(__name__)

# Try to import portalocker for robust file locking
try:
    import portalocker

    _PORTALOCKER_AVAILABLE = True
except ImportError:
    _PORTALOCKER_AVAILABLE = False


class FileLock:
    """Advisory file lock for cross-process synchronization.

    Uses portalocker when available (recommended for production),
    falls back to a simple polling-based lock file mechanism.

    Args:
        lock_path: Path to the lock file.
        timeout: Maximum seconds to wait for the lock.
        poll_interval: Seconds between lock acquisition attempts.
    """

    def __init__(
        self,
        lock_path: str | Path,
        timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> None:
        """Initialize the file lock.

        Args:
            lock_path: Path to the lock file.
            timeout: Maximum wait time for lock acquisition.
            poll_interval: Polling interval for fallback mode.
        """
        self._lock_path = Path(lock_path)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._lock_file: object | None = None

    @contextlib.contextmanager
    def acquire(self) -> Generator[None, None, None]:
        """Acquire the file lock as a context manager.

        Yields:
            None when the lock is held.

        Raises:
            TimeoutError: If the lock cannot be acquired within timeout.
        """
        if _PORTALOCKER_AVAILABLE:
            with self._portalocker_acquire():
                yield
        else:
            with self._fallback_acquire():
                yield

    @contextlib.contextmanager
    def _portalocker_acquire(self) -> Generator[None, None, None]:
        """Acquire lock using portalocker (robust cross-platform)."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "a+") as lock_file:
            try:
                portalocker.lock(
                    lock_file,
                    portalocker.LOCK_EX | portalocker.LOCK_NB,
                )
                logger.debug("file_lock_acquired", path=str(self._lock_path))
                yield
            except portalocker.LockException:
                # Blocking acquire with timeout
                start = time.time()
                while time.time() - start < self._timeout:
                    try:
                        portalocker.lock(
                            lock_file,
                            portalocker.LOCK_EX | portalocker.LOCK_NB,
                        )
                        logger.debug("file_lock_acquired_after_wait", path=str(self._lock_path))
                        yield
                        return
                    except portalocker.LockException:
                        time.sleep(self._poll_interval)
                raise TimeoutError(
                    f"Could not acquire lock on {self._lock_path} within {self._timeout}s"
                ) from None
            finally:
                try:
                    portalocker.unlock(lock_file)
                    logger.debug("file_lock_released", path=str(self._lock_path))
                except Exception:
                    pass

    @contextlib.contextmanager
    def _fallback_acquire(self) -> Generator[None, None, None]:
        """Fallback lock using atomic file creation (works on most systems)."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()

        while time.time() - start < self._timeout:
            try:
                # Try to create lock file atomically
                fd = os.open(
                    str(self._lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                logger.debug("fallback_lock_acquired", path=str(self._lock_path))
                break
            except FileExistsError:
                # Check if lock is stale (process no longer exists)
                if self._is_stale_lock():
                    with contextlib.suppress(FileNotFoundError):
                        self._lock_path.unlink()
                    continue
                time.sleep(self._poll_interval)
        else:
            raise TimeoutError(
                f"Could not acquire lock on {self._lock_path} within {self._timeout}s"
            ) from None

        try:
            yield
        finally:
            with contextlib.suppress(FileNotFoundError):
                self._lock_path.unlink()
                logger.debug("fallback_lock_released", path=str(self._lock_path))

    def _is_stale_lock(self) -> bool:
        """Check if the existing lock file refers to a dead process.

        Returns:
            True if the lock is stale (process no longer exists).
        """
        try:
            pid = int(self._lock_path.read_text().strip())
            # Check if process exists
            if os.name == "nt":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(1, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return False
                return True
            else:
                os.kill(pid, 0)
                return False
        except (ValueError, OSError, ProcessLookupError):
            return True
