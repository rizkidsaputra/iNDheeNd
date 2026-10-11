from pathlib import Path
from collections import defaultdict, Counter
import re


ROOT = Path("data/raw/SPOTIFAI")

assert ROOT.exists(), (
    f"Dataset not found: {ROOT}"
)

configs = defaultdict(set)

complete = 0
incomplete = []

pattern = re.compile(
    r"^pit-size-(\d+)\.txt$"
)

for path in ROOT.rglob("pit-size-*.txt"):
    match = pattern.fullmatch(path.name)

    if not match:
        continue

    run_id = match.group(1)
    folder = path.parent

    relative = folder.relative_to(ROOT)

    configs[str(relative)].add(run_id)

    required = [
        folder / f"rate-trace-{run_id}.txt",
        folder / f"drop-trace-{run_id}.txt",
    ]

    if all(p.exists() for p in required):
        complete += 1
    else:
        incomplete.append(str(path))

print("===== SPOTIFAI INVENTORY =====")

print("Configurations:", len(configs))

print(
    "Runs with PIT trace:",
    sum(map(len, configs.values()))
)

print(
    "Runs with PIT/rate/drop traces:",
    complete
)

print(
    "Incomplete trace groups:",
    len(incomplete)
)

scenario_counts = Counter()

for config, runs in configs.items():
    scenario = Path(config).parts[0]
    scenario_counts[scenario] += len(runs)

print("\n===== SCENARIOS =====")

for scenario, count in sorted(
    scenario_counts.items()
):
    print(f"{scenario}: {count} runs")

print("\n===== CONFIGURATIONS =====")

for config, runs in sorted(
    configs.items()
):
    print(
        f"{config} | runs={len(runs)}"
    )

print("\nINVENTORY COMPLETED")