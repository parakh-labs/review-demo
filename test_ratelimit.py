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
    for kwargs in ({"rate": float("nan"), "capacity": 1}, {"rate": float("inf"), "capacity": 1}, {"rate": -1, "capacity": 1}, {"rate": 0, "capacity": 1}, {"rate": 1, "capacity": float("inf")}, {"rate": 1, "capacity": float("nan")}, {"rate": 1, "capacity": 1, "max_users": float("inf")}, {"rate": 1, "capacity": 1.5}, {"rate": 1, "capacity": True}, {"rate": 1, "capacity": 0}, {"rate": 1, "capacity": 1, "max_users": 0}):
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
    monkeypatch.setattr(ratelimit.TokenBucket, "refill", lambda self, now=None: (scans.append(1), original(self, now))[1])
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


def test_every_access_to_shared_state_happens_under_the_lock():
    from collections import OrderedDict
    from ratelimit import PerUserLimiter

    class CountingLock:
        depth = 0

        def __enter__(self):
            self.depth += 1

        def __exit__(self, *exc):
            self.depth -= 1

    lock = CountingLock()

    class GuardedDict(OrderedDict):
        pass

    def guard(name):
        base = getattr(OrderedDict, name)

        def method(self, *args, **kwargs):
            assert lock.depth > 0, f"buckets.{name} used without the lock"
            return base(self, *args, **kwargs)

        return method

    for name in ("get", "items", "values", "move_to_end", "__len__",
                 "__setitem__", "__delitem__", "__getitem__"):
        setattr(GuardedDict, name, guard(name))

    limiter = PerUserLimiter(rate=1, capacity=2, max_users=1)
    limiter._lock = lock
    limiter.buckets = GuardedDict()
    limiter.allow("a")
    limiter.allow("b")  # saturated: exercises the eviction scan
    limiter.retry_after("a")
    limiter.retry_after("b")  # saturated, unknown id
    assert lock.depth == 0


def test_concurrent_callers_cannot_overspend_one_bucket():
    import sys
    import threading
    from ratelimit import PerUserLimiter
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        limiter = PerUserLimiter(rate=1e-9, capacity=50)
        granted = []

        def worker():
            granted.extend(limiter.allow("a") for _ in range(200))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        sys.setswitchinterval(old)
    assert sum(granted) == 50


def test_deadline_is_measured_from_the_scan_instant_not_a_later_clock_read(monkeypatch):
    import itertools
    import ratelimit
    from ratelimit import PerUserLimiter
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: 0.0)
    limiter = PerUserLimiter(rate=0.1, capacity=2, max_users=1)
    limiter.allow("a", cost=2)
    ticks = itertools.count(1)  # every later clock read is 0.1s later than the last
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: 5.0 + 0.1 * next(ticks))
    # at t=5.0 the bucket holds 0.5 tokens: 15s to full, so deadline is exactly 20.0
    assert limiter._slot_deadline(5.0) == 20.0


def test_non_integer_costs_are_rejected_without_touching_state():
    import pytest
    from ratelimit import PerUserLimiter
    limiter = PerUserLimiter(rate=1, capacity=2)
    for cost in (float("nan"), float("inf"), 1.5, True, "1", None):
        with pytest.raises(ValueError):
            limiter.allow("a", cost=cost)
    assert limiter.buckets == {}
