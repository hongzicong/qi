"""Real-time chunking buffer for RTC-style policy deployment."""

from __future__ import annotations

import threading

import numpy as np


class RealTimeChunkingBuffer:
    """Thread-safe buffer that fuses overlapping action chunks online."""

    def __init__(self, chunk_size: int, exp_weight_factor: float = 0.5, debug: bool = False):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.exp_weight_factor = float(exp_weight_factor)
        self.debug = bool(debug)
        self.control_t = 0
        self.chunks: dict[int, np.ndarray] = {}
        self.generation = 0
        self.lock = threading.Lock()

    def clear(self) -> None:
        """Reset control time, cached action chunks, and invalidate old producers."""
        with self.lock:
            self.control_t = 0
            self.chunks = {}
            self.generation += 1

    def set_control_time(self, control_t: int) -> None:
        """Update the currently executed control step."""
        with self.lock:
            self.control_t = int(control_t)

    def get_control_time(self) -> int:
        """Return the latest control step."""
        with self.lock:
            return self.control_t

    def get_generation(self) -> int:
        """Return the current buffer generation."""
        with self.lock:
            return self.generation

    def has_chunk(self, cursor: int) -> bool:
        """Check whether an action chunk has already been produced for a cursor."""
        with self.lock:
            return int(cursor) in self.chunks

    def enqueue(self, chunk: np.ndarray, cursor: int, generation: int | None = None) -> bool:
        """Insert a model action chunk if it belongs to the current generation."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk shape [T,D], got {chunk.shape}")

        cursor = int(cursor)
        with self.lock:
            if generation is not None and int(generation) != self.generation:
                if self.debug:
                    print(
                        f"[action_chunks] drop stale chunk cursor={cursor} "
                        f"generation={generation} current={self.generation}"
                    )
                return False
            self.chunks[cursor] = chunk
            return True

    def get_action(self, current_time: int) -> np.ndarray | None:
        """Fuse all cached action predictions that cover the current control step."""
        current_time = int(current_time)
        with self.lock:
            relevant = {}
            expired = []
            before_keys = sorted(self.chunks.keys())

            for cursor, chunk in self.chunks.items():
                end = cursor + self.chunk_size
                if cursor <= current_time < end:
                    relevant[cursor] = chunk[current_time - cursor]
                elif end <= current_time:
                    expired.append(cursor)

            for cursor in expired:
                del self.chunks[cursor]

            if self.debug:
                print(
                    f"[action_chunks] t={current_time} before={before_keys} "
                    f"delete={sorted(expired)} after={sorted(self.chunks.keys())}"
                )

            if not relevant:
                return None

            sorted_items = sorted(relevant.items(), key=lambda item: item[0])
            candidate_actions = np.asarray([action for _, action in sorted_items], dtype=np.float32)

        exp_weights = np.exp(self.exp_weight_factor * np.arange(len(candidate_actions), dtype=np.float32))
        exp_weights = (exp_weights / exp_weights.sum())[:, None]
        return (candidate_actions * exp_weights).sum(axis=0).astype(np.float32)