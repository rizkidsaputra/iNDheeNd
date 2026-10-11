from collections import defaultdict
import torch


def simulate_monitoring(
    x,
    times,
    dropped=None,
    delays=None,
):
    """
    x       : [T, N, F] raw features
    times   : consecutive timestamps from one run
    dropped : [T, N] boolean report-loss indicators
    delays  : [T, N] non-negative integer delays

    Returns:
        observed_x : [T, N, F]
        mask       : [T, N]
        age        : [T, N]
        has_report : [T, N]
        source_time: [T, N]
    """
    x = x.detach().cpu().float()
    T, N, F = x.shape

    times = [int(t) for t in times]

    assert len(times) == T
    assert all(
        b == a + 1
        for a, b in zip(
            times[:-1],
            times[1:]
        )
    )

    if dropped is None:
        dropped = torch.zeros(
            T, N, dtype=torch.bool
        )
    else:
        dropped = torch.as_tensor(
            dropped,
            dtype=torch.bool
        )

    if delays is None:
        delays = torch.zeros(
            T, N, dtype=torch.long
        )
    else:
        delays = torch.as_tensor(
            delays,
            dtype=torch.long
        )

    assert dropped.shape == (T, N)
    assert delays.shape == (T, N)
    assert torch.all(delays >= 0)
    assert torch.isfinite(x).all()

    events = defaultdict(list)

    # Schedule transmission events.
    for step, generated_at in enumerate(times):
        for router in range(N):
            if dropped[step, router]:
                continue

            arrival = (
                generated_at + int(delays[step, router])
            )

            events[arrival].append((
                router,
                generated_at,
                x[step, router].clone()
            ))

    # No information available at startup.
    last_features = torch.zeros(N, F)

    last_generated = torch.full(
        (N,),
        -1,
        dtype=torch.long
    )

    observed = []
    masks = []
    ages = []
    availability = []
    source_times = []

    for current_time in times:
        # Process reports that have arrived.
        for (
            router,
            generated_at,
            features
        ) in events[current_time]:

            # Ignore older out-of-order reports.
            if generated_at > int(
                last_generated[router]
            ):
                last_generated[router] = generated_at
                last_features[router] = features

        has_report = last_generated >= 0

        mask = (
            last_generated == current_time
        ).float()

        age = torch.where(
            has_report,
            current_time - last_generated,
            torch.full_like(
                last_generated,
                current_time - times[0] + 1
            )
        ).float()

        observed.append(
            last_features.clone()
        )

        masks.append(mask)
        ages.append(age)

        availability.append(
            has_report.float()
        )

        source_times.append(
            last_generated.clone()
        )

    return {
        "observed_x": torch.stack(observed),
        "mask": torch.stack(masks),
        "age": torch.stack(ages),
        "has_report": torch.stack(availability),
        "source_time": torch.stack(source_times),
    }