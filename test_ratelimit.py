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


def test_saturated_retry_after_is_cached_between_changes(monkeypatch):
    import ratelimit
    from ratelimit import PerUserLimiter
    now = [0.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    limiter = PerUserLimiter(rate=1, capacity=2, max_users=1)
    limiter.allow("a", cost=2)
    scans = []
    original = ratelimit.TokenBucket.refill
    monkeypatch.setattr(ratelimit.TokenBucket, "refill", lambda self: (scans.append(1), original(self))[1])
    for _ in range(5):
        assert limiter.retry_after("b") == 2.0
    assert len(scans) == 1


def test_cost_above_capacity_is_rejected_not_silently_denied():
    import pytest
    from ratelimit import PerUserLimiter
    limiter = PerUserLimiter(rate=1, capacity=2)
    for call in (limiter.allow, limiter.retry_after):
        with pytest.raises(ValueError):
            call("a", cost=3)


def test_rejected_new_ids_do_not_invalidate_the_cached_slot_time(monkeypatch):
    import ratelimit
    from ratelimit import PerUserLimiter
    now = [0.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    limiter = PerUserLimiter(rate=1, capacity=2, max_users=1)
    limiter.allow("a", cost=2)
    limiter.retry_after("b")
    version = limiter._version
    for _ in range(3):
        assert not limiter.allow("b")
    assert limiter._version == version


def test_never_more_permissive_than_unbounded_per_user_buckets(monkeypatch):
    import random
    import ratelimit
    from ratelimit import PerUserLimiter, TokenBucket
    now = [0.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    rng = random.Random(1234)
    bounded = PerUserLimiter(rate=1, capacity=3, max_users=4)
    reference = {}
    for _ in range(3000):
        now[0] += rng.random() * 1.5
        user = f"u{rng.randrange(9)}"
        cost = rng.randint(1, 3)
        if bounded.allow(user, cost):
            ref = reference.setdefault(user, TokenBucket(1, 3))
            assert ref.allow(cost), "bounded limiter allowed what an unbounded one denies"
        else:
            reference.setdefault(user, TokenBucket(1, 3)).refill()
        assert len(bounded.buckets) <= 4
