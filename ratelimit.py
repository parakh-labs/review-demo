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

    At most `max_users` buckets are kept; the least recently used one is
    dropped first, so an attacker cycling ids cannot grow memory without bound.
    """

    def __init__(self, rate: float, capacity: int, max_users: int = 10_000):
        self.rate = rate
        self.capacity = capacity
        self.max_users = max_users
        self.buckets: "OrderedDict[str, TokenBucket]" = OrderedDict()

    def _bucket(self, user_id: str) -> TokenBucket:
        bucket = self.buckets.get(user_id)
        if bucket is None:
            bucket = TokenBucket(self.rate, self.capacity)
            self.buckets[user_id] = bucket
            if len(self.buckets) > self.max_users:
                self.buckets.popitem(last=False)
        else:
            self.buckets.move_to_end(user_id)
        return bucket

    def allow(self, user_id: str, cost: int = 1) -> bool:
        if cost < 1:
            raise ValueError("cost must be at least 1")
        return self._bucket(user_id).allow(cost)

    def retry_after(self, user_id: str, cost: int = 1) -> float:
        """Seconds until `cost` tokens are available; 0 when they are now."""
        bucket = self.buckets.get(user_id)
        if bucket is None:
            return 0.0
        bucket.refill()
        return max(0.0, (cost - bucket.tokens) / bucket.rate)
