from ratelimit import TokenBucket


def test_allows_burst_up_to_capacity():
    bucket = TokenBucket(rate=1, capacity=3)
    assert [bucket.allow() for _ in range(4)] == [True, True, True, False]


def test_users_do_not_share_a_bucket():
    from ratelimit import PerUserLimiter
    limiter = PerUserLimiter(rate=1, capacity=1)
    assert limiter.allow("a") and limiter.allow("b")
