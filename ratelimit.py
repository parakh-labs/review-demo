"""Token-bucket rate limiter used by the API gateway."""
import time
from collections import OrderedDict


class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()

    def refill(self) -> None:
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
        self._no_evict_before = 0.0
        self._version = 0  # bumped whenever a bucket changes
        self._slot_version = -1
        self._slot_at = 0.0

    def _evict_full(self) -> bool:
        """Drop the least recently used bucket that has refilled to capacity.

        A full bucket is indistinguishable from a new one, so dropping it
        cannot hand anyone extra tokens. Returns False if every bucket is
        still draining. A failed scan records the earliest moment any bucket
        can be full (a lower bound: use only drains buckets further), so
        repeated requests while the table is saturated cost O(1), not O(n).
        """
        if time.monotonic() < self._no_evict_before:
            return False
        soonest = float("inf")
        for user_id, bucket in self.buckets.items():
            bucket.refill()
            if bucket.tokens >= bucket.capacity:
                del self.buckets[user_id]
                return True
            soonest = min(soonest, (bucket.capacity - bucket.tokens) / bucket.rate)
        self._no_evict_before = time.monotonic() + soonest
        return False

    def _time_until_slot(self) -> float:
        """Seconds until some bucket is full (0 if one already is).

        Exact, and cached until a bucket is next touched: refill is
        deterministic, so a flood of rejected new ids costs O(1) each.
        """
        now = time.monotonic()
        if self._slot_version != self._version:
            soonest = float("inf")
            for bucket in self.buckets.values():
                bucket.refill()
                soonest = min(soonest, max(0.0, (bucket.capacity - bucket.tokens) / bucket.rate))
            self._slot_at = now + soonest
            self._slot_version = self._version
        return max(0.0, self._slot_at - now)

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

    def allow(self, user_id: str, cost: int = 1) -> bool:
        if cost < 1:
            raise ValueError("cost must be at least 1")
        bucket = self._bucket(user_id)
        self._version += 1
        return bucket is not None and bucket.allow(cost)

    def retry_after(self, user_id: str, cost: int = 1) -> float:
        """Seconds until `cost` tokens are available; 0 when they are now."""
        bucket = self.buckets.get(user_id)
        if bucket is None:
            if len(self.buckets) >= self.max_users:
                # Table saturated: a new id waits until some bucket is full.
                return self._time_until_slot()
            return 0.0
        bucket.refill()
        return max(0.0, (cost - bucket.tokens) / bucket.rate)
