import torch

from monitoring_observation import simulate_monitoring


TIMES = [1, 2, 3, 4]
N = 9
F = 12
ROUT5 = 6

# Unique values at each timestamp/router/feature.
x = torch.arange(
    len(TIMES) * N * F,
    dtype=torch.float32
).reshape(len(TIMES), N, F)


def make_loss():
    return torch.zeros(
        len(TIMES), N, dtype=torch.bool
    )


def make_delay():
    return torch.zeros(
        len(TIMES), N, dtype=torch.long
    )


def test_clean():
    obs = simulate_monitoring(x, TIMES)

    assert torch.equal(obs["observed_x"], x)
    assert torch.all(obs["mask"] == 1)
    assert torch.all(obs["age"] == 0)
    assert torch.all(obs["has_report"] == 1)

    print("TEST 1: Clean monitoring PASSED")


def test_consecutive_loss():
    dropped = make_loss()

    # Rout5 reports t=2 and t=3 are missing.
    dropped[1, ROUT5] = True
    dropped[2, ROUT5] = True

    obs = simulate_monitoring(
        x, TIMES, dropped=dropped
    )

    assert torch.equal(
        obs["observed_x"][1, ROUT5],
        x[0, ROUT5]
    )

    assert torch.equal(
        obs["observed_x"][2, ROUT5],
        x[0, ROUT5]
    )

    assert obs["age"][:, ROUT5].tolist() == [
        0, 1, 2, 0
    ]

    assert obs["mask"][:, ROUT5].tolist() == [
        1, 0, 0, 1
    ]

    print("TEST 2: Consecutive report loss PASSED")


def test_delayed_report():
    dropped = make_loss()
    delays = make_delay()

    # t=2 arrives at t=4.
    delays[1, ROUT5] = 2

    # No newer report can replace it.
    dropped[2, ROUT5] = True
    dropped[3, ROUT5] = True

    obs = simulate_monitoring(
        x,
        TIMES,
        dropped=dropped,
        delays=delays,
    )

    assert obs["source_time"][:, ROUT5].tolist() == [
        1, 1, 1, 2
    ]

    assert torch.equal(
        obs["observed_x"][3, ROUT5],
        x[1, ROUT5]
    )

    # A delayed old report does not count as fresh t=4.
    assert obs["mask"][3, ROUT5] == 0
    assert obs["age"][3, ROUT5] == 2

    print("TEST 3: Delayed report PASSED")


def test_out_of_order():
    dropped = make_loss()
    delays = make_delay()

    # t=2 arrives late at t=4.
    delays[1, ROUT5] = 2

    # t=3 arrives on time.
    dropped[3, ROUT5] = True

    obs = simulate_monitoring(
        x,
        TIMES,
        dropped=dropped,
        delays=delays,
    )

    assert obs["source_time"][2, ROUT5] == 3

    # The older t=2 report arriving at t=4
    # must not overwrite the t=3 report.
    assert obs["source_time"][3, ROUT5] == 3

    assert torch.equal(
        obs["observed_x"][3, ROUT5],
        x[2, ROUT5]
    )

    print("TEST 4: Out-of-order arrival PASSED")


def test_causality():
    dropped = make_loss()
    delays = make_delay()

    delays[1, ROUT5] = 2
    dropped[2, ROUT5] = True
    dropped[3, ROUT5] = True

    obs = simulate_monitoring(
        x,
        TIMES,
        dropped=dropped,
        delays=delays,
    )

    # Report generated at t=2 cannot influence
    # collector state at t=2 or t=3.
    assert torch.equal(
        obs["observed_x"][1, ROUT5],
        x[0, ROUT5]
    )

    assert torch.equal(
        obs["observed_x"][2, ROUT5],
        x[0, ROUT5]
    )

    for step, t in enumerate(TIMES):
        assert torch.all(
            obs["source_time"][step] <= t
        )

    print("TEST 5: Causality PASSED")


def test_empty_collector():
    dropped = make_loss()

    dropped[0, ROUT5] = True
    dropped[1, ROUT5] = True

    obs = simulate_monitoring(
        x,
        TIMES,
        dropped=dropped,
    )

    for t in [0, 1]:
        assert torch.all(
            obs["observed_x"][t, ROUT5] == 0
        )

        assert obs["has_report"][t, ROUT5] == 0
        assert obs["source_time"][t, ROUT5] == -1
        assert obs["mask"][t, ROUT5] == 0

    assert obs["has_report"][2, ROUT5] == 1
    assert obs["source_time"][2, ROUT5] == 3

    print("TEST 6: Empty collector state PASSED")


def main():
    test_clean()
    test_consecutive_loss()
    test_delayed_report()
    test_out_of_order()
    test_causality()
    test_empty_collector()

    print("\nALL MONITORING TESTS PASSED")


if __name__ == "__main__":
    main()