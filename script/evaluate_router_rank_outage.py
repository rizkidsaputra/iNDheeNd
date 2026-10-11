from pathlib import Path

import evaluate_burst_loss as experiment


# Ranking based on PIT increase.
# Router indices follow the validated pilot mapping.
TOP = [
    6,  # Rout5
    1,  # Rout1
    4,  # Rout2
    2,  # Rout9
    7,  # Rout8
    5,  # Rout7
    3,  # Rout6
    0,  # Rout4
    8,  # Rout3
]

BOTTOM = list(reversed(TOP))

# Outage t=45 through t=104.
# IFA begins at t=51.
experiment.SCENARIOS = [
    ("clean", [], 0, 0),

    (
        "top3_60s",
        TOP[:3],
        45,
        104
    ),

    (
        "bottom3_60s",
        BOTTOM[:3],
        45,
        104
    ),

    (
        "top5_60s",
        TOP[:5],
        45,
        104
    ),

    (
        "bottom5_60s",
        BOTTOM[:5],
        45,
        104
    ),

    (
        "top7_60s",
        TOP[:7],
        45,
        104
    ),

    (
        "all9_60s",
        TOP,
        45,
        104
    ),
]

experiment.OUT = Path(
    "results/router_rank_outage_pilot"
)

experiment.OUT.mkdir(
    parents=True,
    exist_ok=True
)

experiment.main()