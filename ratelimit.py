"""Token-bucket rate limiter used by the API gateway."""
import time


class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()

    def allow(self, cost: int = 1) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class PerUserLimiter:
    """One bucket per user id; buckets are created on first use."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.buckets = {}

    def allow(self, user_id: str, cost: int = 1) -> bool:
        bucket = self.buckets.get(user_id)
        if bucket is None:
            bucket = TokenBucket(self.rate, self.capacity)
            self.buckets[user_id] = bucket
        return bucket.allow(cost)

    def retry_after(self, user_id: str) -> float:
        bucket = self.buckets[user_id]
        return (1 - bucket.tokens) / bucket.rate
