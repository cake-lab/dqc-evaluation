import csv
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from qiskit import qasm3
from tabulate import tabulate

from src.config import Config, RunPaths, build_output_namespace

logger = logging.getLogger("dqc_evaluation")


class ExperimentRecorder:
    """Manages incremental recording of experiment results and final persistence.

    Handles CSV writing during the experiment run, and finalizes results as parquet,
    summary.md, and the used config YAML.
    """

    def __init__(
        self,
        run_paths: RunPaths,
        column_names: list[str],
    ):
        """Initialize recorder with prebuilt run paths and CSV column names.

        Args:
            run_paths: The timestamped results directory and derived file paths
            column_names: List of column names for the CSV
        """
        self.paths = run_paths
        self.column_names = column_names
        self.rows = []

        # Open CSV file for incremental writing
        self._csv_handle = self.paths.csv.open(mode="w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_handle, fieldnames=column_names, extrasaction="ignore"
        )
        self._writer.writeheader()

    def _get_qasm_output_path(self, row: dict, circuit_id: str) -> Path:
        """
        Generate the output path for a partitioned circuit in OpenQASM 3.0 format.

        Creates a run-level namespace under the main output directory, then
        preserves the existing circuit path layout.
        Example: output/partitioner-mlfm-r/network-topology-linear/num-qpus-4/disqco/mfpqc/qft/n16/qft_n16_partitioned.qasm

        Args:
            row: Result row containing the namespace fields
            circuit_id: Circuit identifier with dots as separators

        Returns:
            Path object pointing to the QASM file
        """
        parts = circuit_id.split(".")
        namespace = build_output_namespace(row)

        # Create directory path for the run namespace and circuit hierarchy
        dir_path = self.paths.output_root / namespace / Path(*parts[:-1])
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create file name from the last part
        file_name = f"{parts[-1]}_partitioned.qasm"
        return dir_path / file_name

    def _get_result_json_path(self, row: dict, circuit_id: str) -> Path:
        """Return the result JSON path next to the QASM file."""
        parts = circuit_id.split(".")
        namespace = build_output_namespace(row)
        dir_path = self.paths.output_root / namespace / Path(*parts[:-1])
        dir_path.mkdir(parents=True, exist_ok=True)
        file_name = f"{parts[-1]}_result.json"
        return dir_path / file_name

    def _require_nested_result_row(self, row: dict) -> dict:
        """Validate that the row already has the grouped JSON structure."""
        required_keys = {
            "experiment_details",
            "status",
            "input_circuit_metrics",
            "output_circuit_metrics",
            "partitioning_metrics",
        }
        missing = required_keys.difference(row)
        if missing:
            raise TypeError(
                "ExperimentRecorder.record expects a nested row with keys: "
                f"{sorted(required_keys)}; missing {sorted(missing)}"
            )
        return row

    def _flatten_result_row_for_csv(self, row: dict) -> dict:
        """Flatten a grouped row into the existing CSV schema."""
        if "experiment_details" not in row:
            return row

        flat_row: dict[str, object] = {}

        experiment_details = dict(row.get("experiment_details", {}))
        extra_experiment_fields = experiment_details.pop("additional_fields", {})

        flat_row.update(experiment_details)
        if isinstance(extra_experiment_fields, dict):
            flat_row.update(extra_experiment_fields)
        flat_row.update(row.get("status", {}))
        flat_row.update(
            {
                f"input_{key}": value
                for key, value in row.get("input_circuit_metrics", {}).items()
            }
        )
        flat_row.update(
            {
                f"output_{key}": value
                for key, value in row.get("output_circuit_metrics", {}).items()
            }
        )
        flat_row.update(row.get("partitioning_metrics", {}))

        return flat_row

    def record(self, row: dict, partitioned_circuit=None) -> None:
        """Record a single experiment result row.

        Args:
            row: Dictionary with keys matching column_names
            partitioned_circuit: Optional Qiskit QuantumCircuit to save as QASM
        """
        nested_row = self._require_nested_result_row(row)
        flat_row = self._flatten_result_row_for_csv(nested_row)

        self.rows.append(flat_row)
        self._writer.writerow(flat_row)
        self._csv_handle.flush()

        circuit_id = flat_row.get("circuit")

        # Save partitioned circuit as OpenQASM 3.0 if successful and circuit is provided
        if (
            partitioned_circuit is not None
            and nested_row.get("status", {}).get("status") == "success"
        ):
            qasm_path = self._get_qasm_output_path(row, circuit_id)
            with open(qasm_path, "w", encoding="utf-8") as f:
                qasm3.dump(partitioned_circuit, f)

        # Always write a JSON summary for this row (includes status and errors)
        json_path = self._get_result_json_path(flat_row, circuit_id)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(
                nested_row,
                jf,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

    def write_run_metadata(self, config: Config) -> None:
        """Persist run metadata that should exist before experiments begin."""
        self._write_config(config)
        self._write_git_commit(config.git_commit)

    def finalize(self, config: Config) -> None:
        """Finalize results by saving parquet, summary.md, config.yaml, and git metadata.

        Args:
            config: The Config object (provides raw_config for persistence)
        """
        self._csv_handle.close()
        df = pd.DataFrame(self.rows)
        df.to_parquet(self.paths.parquet, index=False)
        self._write_summary(df, config)

    @staticmethod
    def _normalize_status(status: object) -> str:
        """Collapse detailed run statuses into summary-friendly buckets."""
        if isinstance(status, str) and status.startswith("timeout"):
            return "timeout"
        if status is None or pd.isna(status):
            return "unknown"
        return str(status)

    @staticmethod
    def _format_table(headers: list[str], rows: list[list[object]]) -> str:
        if not rows:
            return "_No data available._"

        normalized_rows = [
            ["" if value is None else value for value in row] for row in rows
        ]
        return tabulate(
            normalized_rows, headers=headers, tablefmt="github", disable_numparse=True
        )

    @staticmethod
    def _summarize_values(values: pd.Series, limit: int = 5) -> str:
        unique_values = [str(value) for value in values.dropna().unique()]
        if not unique_values:
            return "-"

        unique_values.sort()
        if len(unique_values) <= limit:
            return ", ".join(unique_values)

        shown = ", ".join(unique_values[:limit])
        return f"{shown}, ... (+{len(unique_values) - limit} more)"

    def _build_overview_rows(
        self, df: pd.DataFrame, config: Config
    ) -> list[list[object]]:
        total_runs = len(df)
        status_series = (
            df["status"].map(self._normalize_status)
            if "status" in df
            else pd.Series(dtype=str)
        )
        status_counts = status_series.value_counts().to_dict()

        successes = int(status_counts.get("success", 0))
        failed = int(status_counts.get("failed", 0))
        skipped = int(status_counts.get("skipped", 0))
        timeouts = int(status_counts.get("timeout", 0))
        partial_success = int(status_counts.get("partial_success", 0))
        other = max(
            total_runs - successes - failed - skipped - timeouts - partial_success, 0
        )
        success_rate = successes / total_runs if total_runs else 0

        partitioning_time = (
            float(df["total_partitioning_exec_time_s"].dropna().sum())
            if "total_partitioning_exec_time_s" in df
            else 0.0
        )
        evaluation_time = (
            float(df["total_evaluation_exec_time_s"].dropna().sum())
            if "total_evaluation_exec_time_s" in df
            else 0.0
        )

        rows: list[list[object]] = [
            ["Results directory", str(self.paths.run_dir)],
            ["Git commit", getattr(config, "git_commit", "unknown")],
        ]

        if getattr(config, "name", None):
            rows.append(["Run name", config.name])
        if getattr(config, "description", None):
            rows.append(["Description", config.description])

        rows.extend(
            [
                ["Runs", total_runs],
                ["Successes", successes],
                ["Failed", failed],
                ["Timeouts", timeouts],
                ["Skipped", skipped],
                ["Partial successes", partial_success],
                ["Other statuses", other],
                ["Success rate", f"{success_rate:.2%}"],
                ["Partitioning time (s)", f"{partitioning_time:.3f}"],
                ["Evaluation time (s)", f"{evaluation_time:.3f}"],
                [
                    "Benchmarks",
                    self._summarize_values(df["benchmark"])
                    if "benchmark" in df
                    else "-",
                ],
                [
                    "Num QPUs",
                    self._summarize_values(df["num_qpus"].astype(str))
                    if "num_qpus" in df
                    else "-",
                ],
                [
                    "Network topologies",
                    self._summarize_values(df["network_topology"])
                    if "network_topology" in df
                    else "-",
                ],
                [
                    "Partitioners",
                    self._summarize_values(df["partitioner"])
                    if "partitioner" in df
                    else "-",
                ],
                ["Timeout setting (s)", getattr(config, "timeout_seconds", "-")],
                [
                    "Skip larger circuits after consecutive timeouts",
                    (
                        str(getattr(config, "skip_larger_circuits_after_timeout", 0))
                        if getattr(config, "skip_larger_circuits_after_timeout", 0)
                        else "disabled"
                    ),
                ],
            ]
        )

        return rows

    def _build_status_rows(self, df: pd.DataFrame) -> list[list[object]]:
        if "status" not in df:
            return []

        normalized = df["status"].map(self._normalize_status)
        total_runs = len(df)
        ordered_statuses = [
            "success",
            "failed",
            "timeout",
            "skipped",
            "partial_success",
        ]
        rows: list[list[object]] = []

        for status in ordered_statuses:
            count = int((normalized == status).sum())
            if count:
                rows.append([status, count, f"{count / total_runs:.2%}"])

        other_count = int((~normalized.isin(ordered_statuses)).sum())
        if other_count:
            rows.append(["other", other_count, f"{other_count / total_runs:.2%}"])

        return rows

    def _build_partitioner_rows(self, df: pd.DataFrame) -> list[list[object]]:
        if "partitioner" not in df or "status" not in df:
            return []

        normalized = df["status"].map(self._normalize_status)
        grouped = (
            pd.DataFrame({"partitioner": df["partitioner"], "status": normalized})
            .groupby("partitioner")["status"]
            .value_counts()
            .unstack(fill_value=0)
        )

        rows: list[list[object]] = []
        for partitioner in sorted(grouped.index.astype(str)):
            counts = grouped.loc[partitioner]
            total = int(counts.sum())
            successes = int(counts.get("success", 0))
            failed = int(counts.get("failed", 0))
            timeouts = int(counts.get("timeout", 0))
            skipped = int(counts.get("skipped", 0))
            partial_success = int(counts.get("partial_success", 0))
            other = max(
                total - successes - failed - timeouts - skipped - partial_success, 0
            )
            rows.append(
                [
                    partitioner,
                    total,
                    successes,
                    failed,
                    timeouts,
                    skipped,
                    partial_success,
                    other,
                    f"{(successes / total if total else 0):.2%}",
                ]
            )

        return rows

    def _build_error_rows(self, df: pd.DataFrame, limit: int = 5) -> list[list[object]]:
        if "error_message" not in df or "status" not in df:
            return []

        mask = df["status"].map(self._normalize_status) != "success"
        errors = df.loc[mask, "error_message"].fillna("Unknown error").astype(str)
        if errors.empty:
            return []

        counts = errors.value_counts().head(limit)
        return [[message, int(count)] for message, count in counts.items()]

    def _write_summary(self, df: pd.DataFrame, config: Config) -> None:
        """Generate and write the experiment summary markdown."""
        summary_lines = ["# Experiment Summary", "", "## Run Overview", ""]
        summary_lines.append(
            self._format_table(
                ["Field", "Value"],
                self._build_overview_rows(df, config),
            )
        )

        status_rows = self._build_status_rows(df)
        if status_rows:
            summary_lines.extend(["", "## Status Breakdown", ""])
            summary_lines.append(
                self._format_table(["Status", "Count", "Share"], status_rows)
            )

        partitioner_rows = self._build_partitioner_rows(df)
        if partitioner_rows:
            summary_lines.extend(["", "## Partitioner Breakdown", ""])
            summary_lines.append(
                self._format_table(
                    [
                        "Partitioner",
                        "Total",
                        "Success",
                        "Failed",
                        "Timeout",
                        "Skipped",
                        "Partial Success",
                        "Other",
                        "Success rate",
                    ],
                    partitioner_rows,
                )
            )

        error_rows = self._build_error_rows(df)
        if error_rows:
            summary_lines.extend(["", "## Frequent Errors", ""])
            summary_lines.append(
                self._format_table(["Error message", "Count"], error_rows)
            )

        summary_text = "\n".join(summary_lines).rstrip() + "\n"

        with self.paths.summary.open(mode="w", encoding="utf-8") as fh:
            fh.write(summary_text)

    def _write_config(self, config: Config) -> None:
        """Save the merged config used for this run."""
        with self.paths.config.open(mode="w", encoding="utf-8") as fh:
            yaml.safe_dump(getattr(config, "raw_config", {}), fh)

    def _write_git_commit(self, git_commit: str) -> None:
        """Save the git commit hash used for this run."""
        with self.paths.git_commit.open(mode="w", encoding="utf-8") as fh:
            fh.write(f"{git_commit}\n")

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        if self.paths.summary.exists():
            with self.paths.summary.open(mode="r", encoding="utf-8") as fh:
                print(fh.read())

    @property
    def run_dir(self) -> Path:
        """Return the run directory path."""
        return self.paths.run_dir

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes CSV file if still open."""
        if hasattr(self._csv_handle, "close"):
            try:
                self._csv_handle.close()
            except OSError as exc:
                logger.warning("Failed to close results CSV: %s", exc)
        return False
