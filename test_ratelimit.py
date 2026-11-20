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


def test_full_table_fails_closed_and_never_resets_a_drained_bucket(monkeypatch):
    import ratelimit
    from ratelimit import PerUserLimiter
    now = [0.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    limiter = PerUserLimiter(rate=1, capacity=2, max_users=2)
    assert limiter.allow("a", cost=2)  # a drained
    assert limiter.allow("b", cost=2)  # b drained
    assert not limiter.allow("c")  # table full of active users: reject
    assert not limiter.allow("a")  # a was not evicted, still drained
    now[0] += 2  # both refill to full
    assert limiter.allow("c")  # a full bucket is evicted to make room
    assert len(limiter.buckets) == 2


def test_cost_is_validated():
    import pytest
    from ratelimit import PerUserLimiter
    with pytest.raises(ValueError):
        PerUserLimiter(rate=1, capacity=1).allow("a", cost=0)


def test_saturated_table_reports_a_retry_delay_and_skips_rescans(monkeypatch):
    import ratelimit
    from ratelimit import PerUserLimiter
    now = [0.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    limiter = PerUserLimiter(rate=1, capacity=2, max_users=1)
    assert limiter.allow("a", cost=2)
    assert limiter.retry_after("b") == 2.0  # asked before any rejected allow()
    assert not limiter.allow("b")
    assert limiter.retry_after("b") == 2.0  # a is full again in 2s
    now[0] += 1
    assert limiter.retry_after("b") == 1.0
    now[0] += 1
    assert limiter.allow("b")  # a refilled, evicted, b admitted


def test_retry_after_is_current_after_existing_users_spend_more(monkeypatch):
    import ratelimit
    from ratelimit import PerUserLimiter
    now = [0.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    limiter = PerUserLimiter(rate=1, capacity=2, max_users=1)
    limiter.allow("a")
    assert not limiter.allow("b")  # caches a lower bound
    limiter.allow("a")  # a drains further
    assert limiter.retry_after("b") == 2.0


def test_invalid_config_is_rejected():
    import pytest
    from ratelimit import PerUserLimiter
    for kwargs in ({"rate": 0, "capacity": 1}, {"rate": 1, "capacity": 0}, {"rate": 1, "capacity": 1, "max_users": 0}):
        with pytest.raises(ValueError):
            PerUserLimiter(**kwargs)
