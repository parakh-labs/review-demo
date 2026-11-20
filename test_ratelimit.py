from ratelimit import TokenBucket


def test_allows_burst_up_to_capacity():
    bucket = TokenBucket(rate=1, capacity=3)
    assert [bucket.allow() for _ in range(4)] == [True, True, True, False]
