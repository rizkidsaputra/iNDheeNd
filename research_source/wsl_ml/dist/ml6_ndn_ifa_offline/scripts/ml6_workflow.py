#!/usr/bin/env python3
"""
ML-6 Offline Inference and Packaging
NDN IFA GraphSAGE + GRU

Commands:
    build
    verify
    predict
    benchmark
    report

No retraining or checkpoint modification.
"""

import argparse
import csv
import hashlib
import json
import platform
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

RELEASE_NAME = "ml6_ndn_ifa_offline"

SOURCE_FILES = [
    "src/ml3_preprocessing.py",
    "src/ml4_observation.py",
    "src/models/graphsage_gru.py",
    "scripts/ml6_workflow.py",
]

VARIANTS = ("plain", "mask", "mask_age")
SEEDS = (42, 52, 62)

ROUTERS = [f"r{i}" for i in range(1, 11)]


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def copy_asset(source, destination):
    source = Path(source)
    destination = Path(destination)
    assert source.is_file(), source
    destination.parent.mkdir(
        parents=True, exist_ok=True
    )
    shutil.copy2(source, destination)
    assert sha256(source) == sha256(destination)


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def setup_imports(package):
    source = str(package / "src")
    if source not in sys.path:
        sys.path.insert(0, source)

    from ml4_observation import MaskAgeEncoder
    from models.graphsage_gru import GraphSAGEGRU

    return MaskAgeEncoder, GraphSAGEGRU


def check_development_results(repo):
    checks = [
        "results/ml4/ml4_validation.json",
        "results/ml5/ml5_validation.json",
        "results/ml5d/ml5d_validation.json",
    ]

    for relative in checks:
        result = read_json(repo / relative)
        assert result["status"] == "PASSED", relative


def find_reference_edges(repo):
    directory = repo / "data/processed/ml2"
    paths = sorted(directory.glob("*.npz"))

    assert len(paths) == 6, (
        "Expected exactly six development NPZ runs"
    )

    reference = None

    for path in paths:
        with np.load(
            path, allow_pickle=False
        ) as d:
            edge = np.asarray(
                d["edge_index"],
                dtype=np.int64
            )

            assert edge.shape == (2, 20)

            assert np.all(edge >= 0)
            assert np.all(edge < 10)

            if reference is None:
                reference = edge.copy()
            else:
                np.testing.assert_array_equal(
                    reference, edge
                )

    return reference


def collect_model_entries(repo, stage):
    entries = []

    ml5a = read_json(
        repo / "results/ml5/checkpoints.json"
    )
    ml5d = read_json(
        repo / "results/ml5d/checkpoints.json"
    )

    assert len(ml5a) == 9
    assert len(ml5d) == 18

    config5a = read_json(
        repo / "configs/ml5/ablation.json"
    )
    config5d = read_json(
        repo / "configs/ml5d/reciprocal.json"
    )

    # ML-5A: one shared train-only encoder
    encoder5a = (
        "results/ml4/encoder.json"
    )

    # ML-5D: separate encoder per training direction
    for item in ml5a:
        variant = item["variant"]
        seed = int(item["seed"])

        assert variant in VARIANTS
        assert seed in SEEDS

        model_id = f"ml5a_{variant}_s{seed}"

        source_ckpt = repo / item["path"]
        source_enc = repo / encoder5a

        assert sha256(source_ckpt) == (
            item["sha256"]
        )

        dest_ckpt = (
            f"models/ml5a/{variant}_seed{seed}.pt"
        )
        dest_enc = "encoders/ml5a/encoder.json"

        copy_asset(
            source_ckpt, stage / dest_ckpt
        )
        copy_asset(
            source_enc, stage / dest_enc
        )

        entries.append({
            "id": model_id,
            "experiment": "ml5a",
            "direction": "original_split",
            "variant": variant,
            "seed": seed,
            "checkpoint": dest_ckpt,
            "checkpoint_sha256":
                sha256(stage / dest_ckpt),
            "encoder": dest_enc,
            "encoder_sha256":
                sha256(stage / dest_enc),
            "train_runs": config5a["train_runs"],
            "replay_runs":
                config5a["holdout_runs"],
            "reference": "ml5a",
        })

    for item in ml5d:
        variant = item["variant"]
        seed = int(item["seed"])
        direction = item["direction"]

        assert variant in VARIANTS
        assert seed in SEEDS
        assert direction in (
            "m0_to_m1", "m1_to_m0"
        )

        model_id = (
            f"ml5d_{direction}_{variant}_s{seed}"
        )

        source_ckpt = repo / item["checkpoint"]

        source_enc = (
            repo / "results/ml5d"
            / direction / "encoder.json"
        )

        assert sha256(source_ckpt) == (
            item["sha256"]
        )

        dest_ckpt = (
            f"models/ml5d/{direction}/"
            f"{variant}_seed{seed}.pt"
        )

        dest_enc = (
            f"encoders/ml5d/{direction}/encoder.json"
        )

        copy_asset(
            source_ckpt, stage / dest_ckpt
        )
        copy_asset(
            source_enc, stage / dest_enc
        )

        split = config5d["directions"][direction]

        entries.append({
            "id": model_id,
            "experiment": "ml5d",
            "direction": direction,
            "variant": variant,
            "seed": seed,
            "checkpoint": dest_ckpt,
            "checkpoint_sha256":
                sha256(stage / dest_ckpt),
            "encoder": dest_enc,
            "encoder_sha256":
                sha256(stage / dest_enc),
            "train_runs": split["train_runs"],
            "replay_runs": split["holdout_runs"],
            "reference": "ml5d",
        })

    entries.sort(key=lambda x: x["id"])

    assert len(entries) == 27

    ids = [x["id"] for x in entries]
    assert len(set(ids)) == len(ids)

    # Validate checkpoint-to-encoder provenance.
    for item in entries:
        checkpoint = torch.load(
            stage / item["checkpoint"],
            map_location="cpu",
            weights_only=True,
        )

        assert checkpoint["variant"] == (
            item["variant"]
        )
        assert int(checkpoint["seed"]) == (
            item["seed"]
        )

        assert checkpoint["train_runs"] == (
            item["train_runs"]
        )

        assert checkpoint["encoder_sha256"] == (
            item["encoder_sha256"]
        )

        if item["experiment"] == "ml5d":
            assert checkpoint["direction"] == (
                item["direction"]
            )

        cfg = checkpoint["model_config"]

        assert cfg["input_dim"] == 13
        assert cfg["hidden_dim"] == 64
        assert cfg["gru_hidden_dim"] == 64

        encoder = read_json(
            stage / item["encoder"]
        )

        assert encoder["train_runs"] == (
            item["train_runs"]
        )

        assert encoder["input_dim"] == 13

    return entries


def make_readme(stage):
    text = """# NDN IFA — ML-6 Offline Inference Package

This is an experimental inference package.

## Requirements

Python 3, PyTorch, PyTorch Geometric, NumPy.
Use compatible installed versions of PyTorch and PyG.

See `environment.json` for the build environment.

## Input

An NPZ file with five arrays:

- x: [B, 5, 10, 10]
- mask: [B, 5, 10]
- age: [B, 5, 10]
- age_valid: [B, 5, 10]
- feature_valid: [B, 5, 10, 10]

All tensors must use router order r1 ... r10.

The package uses a fixed topology stored in:
`assets/edge_index.npy`.

The input may additionally contain `edge_index`;
if supplied, it must match the packaged topology.

Raw NFD cumulative counters must be processed
causally before using this package.

No ground-truth labels or future reports are
required for inference.

## List model IDs

python scripts/ml6_workflow.py list

## Predict

python scripts/ml6_workflow.py predict \\
  --id ml5a_mask_s42 \\
  --input /path/to/window.npz \\
  --output /path/to/prediction.json

## Interpretation

0 = Normal
1 = IFA

The threshold is fixed at 0.5.

The output is experimental, not evidence of
production robustness.

## Important

- Do not mix checkpoints and encoders.
- Do not change router ordering.
- Do not normalize telemetry twice.
- Missing feature values must follow the
  original ML-2 zero-fill convention.
- Age is actual causal observation freshness.
- Do not substitute synthetic nominal Age
  in operational inference.
- Monitoring collection and deadline enforcement
  are outside this offline package.
"""
    (stage / "README.md").write_text(
        text, encoding="utf-8"
    )


def build_package(repo, destination):
    check_development_results(repo)

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    stage = Path(tempfile.mkdtemp(
        prefix=".ml6_stage_",
        dir=destination.parent,
    ))

    try:
        for relative in SOURCE_FILES:
            copy_asset(
                repo / relative,
                stage / relative,
            )

        models_init = stage / "src/models/__init__.py"
        if not models_init.exists():
            models_init.write_text(
                "", encoding="utf-8"
            )

        for relative in [
            "configs/ml5/ablation.json",
            "configs/ml5d/reciprocal.json",
        ]:
            copy_asset(
                repo / relative,
                stage / relative,
            )

        optional_schema = (
            repo / "results/ml1/feature_schema.json"
        )

        if optional_schema.is_file():
            copy_asset(
                optional_schema,
                stage / "assets/feature_schema.json",
            )

        edge = find_reference_edges(repo)

        asset_path = (
            stage / "assets/edge_index.npy"
        )
        asset_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        np.save(asset_path, edge)

        entries = collect_model_entries(
            repo, stage
        )

        import torch_geometric

        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric":
                torch_geometric.__version__,
            "numpy": np.__version__,
            "torch_cuda_version":
                torch.version.cuda,
            "cuda_available":
                torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }

        save_json(
            stage / "environment.json",
            environment
        )

        make_readme(stage)

        runtime_hashes = {
            relative: sha256(stage / relative)
            for relative in SOURCE_FILES
        }

        manifest = {
            "schema_version": 1,
            "name": "ndn_ifa_ml6_offline",
            "status": "experimental",
            "threshold": 0.5,
            "label_mapping": {
                "0": "Normal",
                "1": "IFA",
            },
            "router_order": ROUTERS,
            "window_steps": 5,
            "interval_seconds": 3,
            "node_count": 10,
            "feature_count": 10,
            "encoded_channels": 13,
            "edge_index_path":
                "assets/edge_index.npy",
            "edge_index_sha256":
                sha256(asset_path),
            "runtime_sha256": runtime_hashes,
            "models": entries,
            "limitations": [
                "Development-only validation",
                "Not proven robust to monitoring loss",
                "M0/M1 protocol confounding",
                "No independent final test",
                "No online collector integration",
            ],
        }

        save_json(
            stage / "registry.json",
            manifest
        )

        # Replace only the ML-6 distribution folder.
        if destination.exists():
            shutil.rmtree(destination)

        stage.rename(destination)

        print("=== ML-6 BUILD PASSED ===")
        print("Package:", destination)
        print("Registered models:", len(entries))
        print("Topology:", tuple(edge.shape))
        print("Registry:", destination / "registry.json")

    except Exception:
        shutil.rmtree(
            stage, ignore_errors=True
        )
        raise


class OfflineInference:
    def __init__(
        self,
        package,
        model_id,
        device="cpu",
    ):
        self.package = Path(package).resolve()

        registry_path = (
            self.package / "registry.json"
        )

        self.registry = read_json(
            registry_path
        )

        assert self.registry["schema_version"] == 1
        assert self.registry["threshold"] == 0.5
        assert self.registry["router_order"] == ROUTERS
        assert self.registry["window_steps"] == 5

        # Validate packaged source integrity.
        for relative, digest in (
            self.registry["runtime_sha256"].items()
        ):
            assert sha256(
                self.package / relative
            ) == digest, (
                f"Runtime integrity failure: {relative}"
            )

        matches = [
            x for x in self.registry["models"]
            if x["id"] == model_id
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Unknown model ID: {model_id}"
            )

        self.entry = matches[0]

        ckpt_path = (
            self.package
            / self.entry["checkpoint"]
        )

        encoder_path = (
            self.package
            / self.entry["encoder"]
        )

        assert sha256(ckpt_path) == (
            self.entry["checkpoint_sha256"]
        )

        assert sha256(encoder_path) == (
            self.entry["encoder_sha256"]
        )

        edge_path = (
            self.package
            / self.registry["edge_index_path"]
        )

        assert sha256(edge_path) == (
            self.registry["edge_index_sha256"]
        )

        edges = np.load(
            edge_path, allow_pickle=False
        )

        assert edges.shape == (2, 20)

        self.device = torch.device(device)
        self.edge_numpy = edges.copy()
        self.edges = torch.from_numpy(
            edges.astype(np.int64)
        ).to(self.device)

        MaskAgeEncoder, GraphSAGEGRU = (
            setup_imports(self.package)
        )

        self.encoder = MaskAgeEncoder.load(
            encoder_path
        )

        assert self.encoder.train_runs == (
            self.entry["train_runs"]
        )

        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=True,
        )

        assert checkpoint["encoder_sha256"] == (
            self.entry["encoder_sha256"]
        )

        assert checkpoint["variant"] == (
            self.entry["variant"]
        )

        assert checkpoint["train_runs"] == (
            self.entry["train_runs"]
        )

        assert checkpoint["model_config"]["input_dim"] == 13

        self.model = GraphSAGEGRU(
            **checkpoint["model_config"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        assert sum(
            p.numel()
            for p in self.model.parameters()
        ) == 35009

        self.model = self.model.to(
            self.device
        ).eval()

        self.threshold = 0.5

    def read_input(self, path):
        """
        Read ML-2 temporal windows or standalone
        ML-6 inference arrays.

        ML-2:
          X_window             [46,5,10,10]
          mask_window          [46,5,10]
          age_window           [46,5,10]
          age_valid_window     [46,5,10]
          feature_valid_window [46,5,10,10]

        ML-6:
          x             [B,5,10,10]
          mask          [B,5,10]
          age           [B,5,10]
          age_valid     [B,5,10]
          feature_valid [B,5,10,10]

        Do not mix timestep arrays with
        window-level arrays.
        """

        with np.load(
            path,
            allow_pickle=False
        ) as d:

            if "X_window" in d.files:
                # ML-2 full run file.
                mapping = {
                    "x": "X_window",
                    "mask": "mask_window",
                    "age": "age_window",
                    "age_valid": "age_valid_window",
                    "feature_valid": "feature_valid_window",
                }

            elif "x" in d.files:
                # Standalone ML-6 inference input.
                mapping = {
                    "x": "x",
                    "mask": "mask",
                    "age": "age",
                    "age_valid": "age_valid",
                    "feature_valid": "feature_valid",
                }

            else:
                raise ValueError(
                    "Unsupported inference input format"
                )

            arrays = {}

            for name, source_key in mapping.items():
                if source_key not in d.files:
                    raise KeyError(
                        f"Missing required array: {source_key}"
                    )

                arrays[name] = np.asarray(
                    d[source_key]
                ).copy()

            if "edge_index" in d.files:
                np.testing.assert_array_equal(
                    d["edge_index"],
                    self.edge_numpy,
                )

        return arrays

    def validate_and_prepare(self, arrays):
        required = (
            "x", "mask", "age",
            "age_valid", "feature_valid"
        )

        if set(arrays.keys()) != set(required):
            raise ValueError(
                "Input keys must contain only "
                "the five ML-6 observation arrays"
            )

        x = np.asarray(
            arrays["x"], dtype=np.float32
        )

        # Support one window without B dimension.
        if x.ndim == 3:
            x = x[None, ...]

        if x.ndim != 4:
            raise ValueError(
                "Expected x [B,5,10,10]"
            )

        B, T, N, F = x.shape

        if (T, N, F) != (5, 10, 10):
            raise ValueError(
                f"Unexpected X shape: {x.shape}"
            )

        if B < 1:
            raise ValueError("Empty input batch")

        result = {"x": x}

        for name in [
            "mask", "age", "age_valid"
        ]:
            array = np.asarray(
                arrays[name],
                dtype=np.float32
            )

            if array.ndim == 2:
                array = array[None, ...]

            if array.shape != (B, T, N):
                raise ValueError(
                    f"Invalid {name} shape: "
                    f"{array.shape}"
                )

            result[name] = array

        fv = np.asarray(
            arrays["feature_valid"],
            dtype=np.float32,
        )

        if fv.ndim == 3:
            fv = fv[None, ...]

        if fv.shape != (B, T, N, F):
            raise ValueError(
                "Invalid feature_valid shape"
            )

        result["feature_valid"] = fv

        for name, array in result.items():
            if not np.isfinite(array).all():
                raise ValueError(
                    f"Nonfinite {name}"
                )

        for name in [
            "mask", "age_valid",
            "feature_valid",
        ]:
            array = result[name]

            if not np.isin(
                array, [0.0, 1.0]
            ).all():
                raise ValueError(
                    f"Invalid binary {name}"
                )

        if np.any(result["age"] < 0):
            raise ValueError(
                "Age cannot be negative"
            )

        if np.any(
            result["feature_valid"]
            > result["mask"][..., None]
        ):
            raise ValueError(
                "Feature valid despite missing report"
            )

        if np.any(
            result["x"][
                result["feature_valid"] == 0
            ] != 0
        ):
            raise ValueError(
                "Invalid feature must be zero-filled"
            )

        if np.any(
            result["age"][
                result["age_valid"] == 0
            ] != 0
        ):
            raise ValueError(
                "Invalid Age must be zero-filled"
            )

        return {
            key: torch.from_numpy(
                np.ascontiguousarray(value)
            ).to(self.device)
            for key, value in result.items()
        }

    @torch.inference_mode()
    def predict(self, arrays):
        batch = self.validate_and_prepare(
            arrays
        )

        features = self.encoder.encode(
            batch,
            self.entry["variant"]
        )

        assert features.shape[-1] == 13

        logits = self.model(
            features, self.edges
        )

        probabilities = torch.sigmoid(
            logits
        ).detach().cpu().numpy()

        assert np.isfinite(
            probabilities
        ).all()

        predictions = (
            probabilities >= self.threshold
        ).astype(np.int64)

        return probabilities, predictions


def model_list(package):
    registry = read_json(
        package / "registry.json"
    )

    print("=== ML-6 MODEL REGISTRY ===")
    print("Package:", package)

    for item in registry["models"]:
        print(
            item["id"],
            "|", item["variant"],
            "|", item["direction"]
        )

    print("Total:", len(registry["models"]))


def predict_cli(args, package):
    runner = OfflineInference(
        package, args.id, args.device
    )

    arrays = runner.read_input(
        args.input
    )

    probabilities, predictions = (
        runner.predict(arrays)
    )

    records = []

    for i, (prob, pred) in enumerate(
        zip(probabilities, predictions)
    ):
        records.append({
            "sample_index": i,
            "probability_ifa": float(prob),
            "prediction": int(pred),
            "class_name": (
                "IFA" if pred else "Normal"
            ),
        })

    output = {
        "model_id": args.id,
        "device": args.device,
        "threshold": runner.threshold,
        "samples": records,
    }

    save_json(args.output, output)

    print("=== ML-6 OFFLINE INFERENCE ===")
    print("Model:", args.id)
    print("Input windows:", len(records))
    print("Predicted IFA:", sum(
        r["prediction"] for r in records
    ))
    print("Output:", args.output)

    if len(records) <= 5:
        for r in records:
            print(r)


def reference_predictions(repo, entry, run_id):
    if entry["reference"] == "ml5a":
        path = (
            repo / "results/ml5/all_predictions.csv"
        )
    else:
        path = (
            repo / "results/ml5d/all_predictions.csv"
        )

    selected = []

    with path.open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            if row["run_id"] != run_id:
                continue

            if row["variant"] != (
                entry["variant"]
            ):
                continue

            if int(row["seed"]) != (
                entry["seed"]
            ):
                continue

            if row["split"] != "holdout":
                continue

            if entry["reference"] == "ml5d":
                if row["direction"] != (
                    entry["direction"]
                ):
                    continue

                if row["condition"] != "original":
                    continue

            selected.append(row)

    selected.sort(
        key=lambda r: int(
            r["end_sequence"]
        )
    )

    assert len(selected) == 46

    return selected


def verify_cli(repo, package, device):
    registry = read_json(
        package / "registry.json"
    )

    entries = registry["models"]
    assert len(entries) == 27

    results = []
    max_error_global = 0.0
    comparisons = 0

    print("=== ML-6 REPLAY VERIFICATION ===")
    print("Device:", device)

    for entry in entries:
        runner = OfflineInference(
            package, entry["id"], device
        )

        for run_id in entry["replay_runs"]:
            path = (
                repo / "data/processed/ml2"
                / f"{run_id}.npz"
            )

            input_arrays = runner.read_input(path)

            probabilities, predictions = (
                runner.predict(input_arrays)
            )

            references = reference_predictions(
                repo, entry, run_id
            )

            # Ground truth read ONLY in verification,
            # never inside the inference runtime.
            with np.load(
                path, allow_pickle=False
            ) as d:
                labels = d["y_window"]

            assert len(probabilities) == 46
            assert len(labels) == 46

            expected = np.array([
                float(r["probability"])
                for r in references
            ])

            expected_predictions = np.array([
                int(r["prediction"])
                for r in references
            ])

            expected_labels = np.array([
                int(r["label"])
                for r in references
            ])

            sequences = np.array([
                int(r["end_sequence"])
                for r in references
            ])

            np.testing.assert_array_equal(
                sequences, np.arange(5, 51)
            )

            np.testing.assert_array_equal(
                labels, expected_labels
            )

            np.testing.assert_allclose(
                probabilities, expected,
                rtol=1e-5,
                atol=1e-5,
            )

            np.testing.assert_array_equal(
                predictions,
                expected_predictions,
            )

            error = float(np.max(
                np.abs(probabilities - expected)
            ))

            max_error_global = max(
                max_error_global, error
            )

            comparisons += len(probabilities)

            results.append({
                "model_id": entry["id"],
                "run_id": run_id,
                "windows": len(probabilities),
                "max_abs_error": error,
                "status": "PASSED",
            })

        print(
            "[PASS]", entry["id"],
            "| 92 predictions reproduced"
        )

    assert len(results) == 54
    assert comparisons == 2484

    output = repo / "results/ml6"

    write_csv(
        output / "replay_audit.csv",
        results,
        [
            "model_id", "run_id",
            "windows", "max_abs_error",
            "status",
        ]
    )

    report = {
        "status": "PASSED",
        "checkpoints_verified": 27,
        "run_replays": len(results),
        "probabilities_compared": comparisons,
        "max_abs_error":
            max_error_global,
        "device": device,
        "limitations": [
            "Replay on previously examined development data",
            "Not independent robustness validation",
        ],
    }

    save_json(
        output / "ml6_validation.json",
        report
    )

    print("\n==============================")
    print("ML-6 REPLAY VALIDATION: PASSED")
    print("Models:", 27)
    print("Predictions compared:", comparisons)
    print(
        "Maximum probability difference:",
        max_error_global
    )
    print("==============================")


def create_sample(repo):
    source = (
        repo / "data/processed/ml2"
        / "m1dev_v16_r02_a1.npz"
    )

    output = (
        repo / "results/ml6/sample_input.npz"
    )

    # Save only observation tensors.
    # Exclude labels and source-audit data.
    with np.load(
        source, allow_pickle=False
    ) as d:
        arrays = {
            "x": d["X_window"][:1],
            "mask": d["mask_window"][:1],
            "age": d["age_window"][:1],
            "age_valid":
                d["age_valid_window"][:1],
            "feature_valid":
                d["feature_valid_window"][:1],
        }

    np.savez_compressed(
        output, **arrays
    )

    return output


def benchmark_cli(repo, package):
    import statistics

    results = []
    sample = create_sample(repo)

    model_ids = [
        f"ml5a_{v}_s42"
        for v in VARIANTS
    ] + [
        f"ml5d_m0_to_m1_{v}_s42"
        for v in VARIANTS
    ]

    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    print("=== ML-6 LATENCY BENCHMARK ===")

    for device in devices:
        for model_id in model_ids:
            runner = OfflineInference(
                package, model_id, device
            )

            arrays = runner.read_input(
                sample
            )

            def synchronize():
                if device == "cuda":
                    torch.cuda.synchronize()

            # Warmup excluded from latency results.
            for _ in range(5):
                runner.predict(arrays)

            synchronize()

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            latencies = []

            # End-to-end offline model call,
            # including encoder preprocessing.
            for _ in range(30):
                synchronize()
                start = time.perf_counter()

                runner.predict(arrays)

                synchronize()

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(elapsed_ms)

            p50 = float(np.percentile(
                latencies, 50
            ))

            p95 = float(np.percentile(
                latencies, 95
            ))

            mean = statistics.mean(
                latencies
            )

            gpu_mb = (
                torch.cuda.max_memory_allocated()
                / (1024 * 1024)
                if device == "cuda"
                else None
            )

            # Linux ru_maxrss is process-level peak KB.
            rss_mb = (
                resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss / 1024.0
            )

            row = {
                "model_id": model_id,
                "device": device,
                "windows_per_call": 1,
                "repeats": 30,
                "mean_ms": mean,
                "p50_ms": p50,
                "p95_ms": p95,
                "process_peak_rss_mb":
                    rss_mb,
                "gpu_peak_allocated_mb":
                    gpu_mb,
            }

            results.append(row)

            print(
                f"[PASS] {model_id} "
                f"{device}: "
                f"p50={p50:.3f} ms, "
                f"p95={p95:.3f} ms"
            )

    write_csv(
        repo / "results/ml6/latency_benchmark.csv",
        results,
        [
            "model_id", "device",
            "windows_per_call", "repeats",
            "mean_ms", "p50_ms", "p95_ms",
            "process_peak_rss_mb",
            "gpu_peak_allocated_mb",
        ]
    )

    print("ML-6 BENCHMARK COMPLETED")


def report_cli(repo, package):
    validation = read_json(
        repo / "results/ml6/ml6_validation.json"
    )

    assert validation["status"] == "PASSED"

    registry = read_json(
        package / "registry.json"
    )

    benchmark = (
        repo / "results/ml6/latency_benchmark.csv"
    )

    assert benchmark.is_file()

    lines = [
        "# ML-6 — Offline Inference & Model Packaging",
        "",
        "## Status",
        "",
        "**Implementation and replay: PASSED**",
        "",
        "## Model Registry",
        "",
        f"- Registered models: "
        f"{len(registry['models'])}",
        "- ML-5A checkpoints: 9",
        "- ML-5D checkpoints: 18",
        "- Input features: 10 NFD metrics",
        "- Encoded input channels: 13",
        "- Temporal window: 5 steps",
        "- Routers: 10",
        "- Directed edges: 20",
        "- Decision threshold: 0.5",
        "",
        "## Reproducibility",
        "",
        f"- Predictions compared: "
        f"{validation['probabilities_compared']}",
        f"- Maximum probability difference: "
        f"{validation['max_abs_error']:.10f}",
        "- Checkpoint hashes verified",
        "- Model/encoder provenance verified",
        "- Router order and topology fixed",
        "",
        "## Offline Benchmark",
        "",
        "Latency results: `latency_benchmark.csv`.",
        "",
        "Benchmark measures one-window inference",
        "including encoder preprocessing and",
        "tensor transfers inside the runtime call.",
        "",
        "It does not include collector acquisition,",
        "NFD report parsing, or network delay.",
        "",
        "Process RSS reports a cumulative process peak,",
        "not exclusive memory usage of one model.",
        "",
        "## Distribution",
        "",
        f"- Package: `{RELEASE_NAME}`",
        "- Registry: `registry.json`",
        "- Model checkpoints: `models/`",
        "- Encoders: `encoders/`",
        "- Frozen graph: `assets/edge_index.npy`",
        "- Inference script: `scripts/ml6_workflow.py`",
        "- Environment: `environment.json`",
        "",
        "## Scientific Limitations",
        "",
        "- No retraining was performed.",
        "- Replay uses previously examined data.",
        "- Mask improvement is not yet established.",
        "- Mask+Age is sensitive to distribution shift.",
        "- Robustness is not yet independently validated.",
        "- ML-7 must implement causal online windowing.",
        "- ML-8 requires independent network runs.",
        "",
        "## Next Milestone",
        "",
        "ML-7: Collector-side integration.",
    ]

    path = (
        repo / "results/ml6/ml6_report.md"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("ML-6 report saved:", path)


def main():
    parser = argparse.ArgumentParser(
        description="NDN IFA ML-6 offline workflow"
    )

    sub = parser.add_subparsers(
        dest="command", required=True
    )

    build = sub.add_parser("build")
    build.add_argument("--output")

    verify = sub.add_parser("verify")
    verify.add_argument("--package")
    verify.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu"
    )

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--package")

    predict = sub.add_parser("predict")
    predict.add_argument("--package")
    predict.add_argument(
        "--id", required=True
    )
    predict.add_argument(
        "--input", required=True
    )
    predict.add_argument(
        "--output", required=True
    )
    predict.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu"
    )

    listing = sub.add_parser("list")
    listing.add_argument("--package")

    report = sub.add_parser("report")
    report.add_argument("--package")

    args = parser.parse_args()

    embedded_package = (
        ROOT if (ROOT / "registry.json").is_file()
        else ROOT / "dist" / RELEASE_NAME
    )

    package = Path(
        getattr(args, "package", None)
        or embedded_package
    ).resolve()

    if args.command == "build":
        destination = Path(
            args.output
            or ROOT / "dist" / RELEASE_NAME
        ).resolve()

        build_package(ROOT, destination)

    elif args.command == "list":
        model_list(package)

    elif args.command == "predict":
        predict_cli(args, package)

    elif args.command == "verify":
        verify_cli(
            ROOT, package, args.device
        )

    elif args.command == "benchmark":
        benchmark_cli(
            ROOT, package
        )

    elif args.command == "report":
        report_cli(
            ROOT, package
        )


if __name__ == "__main__":
    main()
