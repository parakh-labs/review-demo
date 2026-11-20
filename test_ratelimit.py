from ratelimit import TokenBucket


def test_allows_burst_up_to_capacity():
    bucket = TokenBucket(rate=1, capacity=3)
    assert [bucket.allow() for _ in range(4)] == [True, True, True, False]


def test_users_do_not_share_a_bucket():
    from ratelimit import PerUserLimiter
    limiter = PerUserLimiter(rate=0.001, capacity=1)
    assert limiter.allow("a")
    assert not limiter.allow("a")  # a's own bucket is empty
    assert limiter.allow("b")  # b is unaffected


def test_retry_after_counts_elapsed_time_and_never_goes_negative(monkeypatch):
    import ratelimit
    from ratelimit import PerUserLimiter
    now = [100.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    limiter = PerUserLimiter(rate=1, capacity=1)
    assert limiter.retry_after("nobody") == 0.0
    limiter.allow("a")
    assert limiter.retry_after("a") == 1.0
    now[0] += 0.5
    assert limiter.retry_after("a") == 0.5
    now[0] += 10
    assert limiter.retry_after("a") == 0.0


def test_buckets_are_bounded_and_cost_is_validated():
    import pytest
    from ratelimit import PerUserLimiter
    limiter = PerUserLimiter(rate=1, capacity=1, max_users=2)
    for user in ("a", "b", "c"):
        limiter.allow(user)
    assert list(limiter.buckets) == ["b", "c"]
    with pytest.raises(ValueError):
        limiter.allow("a", cost=0)
