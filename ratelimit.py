"""Token-bucket rate limiter used by the API gateway."""
import threading
import time
from collections import OrderedDict


class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()

    def refill(self, now: float = None) -> None:
        if now is None:
            now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def allow(self, cost: int = 1) -> bool:
        self.refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class PerUserLimiter:
    """One bucket per user id; buckets are created on first use.

    At most `max_users` buckets are kept. Only buckets that have refilled to
    capacity are evicted; when none can be, new ids are rejected (fail closed),
    so cycling ids can neither grow memory nor reset a drained bucket.
    """

    def __init__(self, rate: float, capacity: int, max_users: int = 10_000):
        if rate <= 0 or capacity < 1 or max_users < 1:
            raise ValueError("rate must be > 0, capacity and max_users >= 1")
        self.rate = rate
        self.capacity = capacity
        self.max_users = max_users
        self.buckets: "OrderedDict[str, TokenBucket]" = OrderedDict()
        self._lock = threading.RLock()
        self._version = 0  # bumped whenever a bucket changes
        self._slot_version = -1
        self._slot_at = 0.0

    def _scan(self, now: float):
        """One pass at a single instant `now`.

        Returns (user id of the least recently used full bucket or None,
        seconds until some bucket is full). The deadline is cached until a
        bucket is next touched: refill is deterministic, so while the table
        is saturated repeated callers cost O(1), not O(n).
        """
        soonest = float("inf")
        for user_id, bucket in self.buckets.items():
            bucket.refill(now)
            remaining = max(0.0, (bucket.capacity - bucket.tokens) / bucket.rate)
            if remaining == 0.0:
                return user_id, 0.0
            soonest = min(soonest, remaining)
        return None, soonest

    def _slot_deadline(self, now: float) -> float:
        """Absolute time at which some bucket is full (<= now if one is)."""
        if self._slot_version != self._version:
            full, soonest = self._scan(now)
            self._slot_at = now if full is not None else now + soonest
            self._slot_version = self._version
        return self._slot_at

    def _evict_full(self) -> bool:
        """Drop the least recently used bucket that has refilled to capacity.

        A full bucket is indistinguishable from a new one, so dropping it
        cannot hand anyone extra tokens. Returns False if every bucket is
        still draining.
        """
        now = time.monotonic()
        if self._slot_version == self._version and now < self._slot_at:
            return False
        full, soonest = self._scan(now)
        if full is None:
            self._slot_at, self._slot_version = now + soonest, self._version
            return False
        del self.buckets[full]
        return True

    def _time_until_slot(self) -> float:
        """Seconds until some bucket is full (0 if one already is)."""
        now = time.monotonic()
        return max(0.0, self._slot_deadline(now) - now)

    def _bucket(self, user_id: str):
        bucket = self.buckets.get(user_id)
        if bucket is not None:
            self.buckets.move_to_end(user_id)
            return bucket
        if len(self.buckets) >= self.max_users and not self._evict_full():
            return None  # table full of active users: fail closed
        bucket = TokenBucket(self.rate, self.capacity)
        self.buckets[user_id] = bucket
        return bucket

    def _check_cost(self, cost: int) -> None:
        if cost < 1 or cost > self.capacity:
            raise ValueError(f"cost must be between 1 and capacity ({self.capacity})")

    def allow(self, user_id: str, cost: int = 1) -> bool:
        self._check_cost(cost)
        with self._lock:
            bucket = self._bucket(user_id)
            if bucket is None:
                return False  # nothing changed, so the cached slot time stays valid
            self._version += 1
            return bucket.allow(cost)

    def retry_after(self, user_id: str, cost: int = 1) -> float:
        """Seconds until `cost` tokens are available; 0 when they are now."""
        self._check_cost(cost)
        with self._lock:
            bucket = self.buckets.get(user_id)
            if bucket is None:
                if len(self.buckets) >= self.max_users:
                    # Table saturated: a new id waits until some bucket is full.
                    return self._time_until_slot()
                return 0.0
            bucket.refill()
            return max(0.0, (cost - bucket.tokens) / bucket.rate)
