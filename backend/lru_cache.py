"""LRU Cache for storing pipeline responses."""

import threading
from collections import OrderedDict
from typing import Any


class SimplePipelineCache:
    """Thread-safe LRU Cache for pipeline events."""
    
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.cache: OrderedDict[str, list[tuple[str, Any]]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, query: str) -> list[tuple[str, Any]] | None:
        """Get the cached event list for a query, if present."""
        query = query.strip().lower()
        with self.lock:
            if query in self.cache:
                self.cache.move_to_end(query)
                return self.cache[query]
            return None

    def put(self, query: str, events: list[tuple[str, Any]]) -> None:
        """Store the event list for a query, evicting the oldest if full."""
        query = query.strip().lower()
        with self.lock:
            self.cache[query] = events
            self.cache.move_to_end(query)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


# Global singleton instance
pipeline_cache = SimplePipelineCache(capacity=20)
