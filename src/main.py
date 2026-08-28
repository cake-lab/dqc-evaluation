import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from qceval import CircuitLoader, CircuitMetricsEvaluator
from tqdm import tqdm

from src import setup_logging
from src.config import Config, build_metric_ids, load_config
from src.metrics import CUSTOM_METRIC_REGISTRY
from src.network import build_network_topology
from src.partitioner.outcome import PARTITIONER_DISTRIBUTION_COST_METRIC
from src.results import ExperimentRecorder, ExperimentResultRow


@dataclass
class CircuitProcessingResult:
    row: ExperimentResultRow
    partitioned_circuit: object | None = None
    timeout_qubits: int | None = None
    skip_current_combo: bool = False


def _compact_error_message(message: object, fallback: str) -> str:
    """Return a short single-line error message for CSV/result storage."""
    if not isinstance(message, str):
        return fallback

    compact_message = message.strip()
    if not compact_message:
        return fallback

    return compact_message.splitlines()[-1].strip() or fallback


def _handle_keyboard_interrupt(logger) -> bool:
    """Handle a keyboard interrupt by asking the user whether to skip the current test or cancel the entire experiment."""
    prompt = (
        "\nKeyboard interrupt received."
        " Skip this test and continue (s) or cancel experiment (c)? [s/c]: "
    )
    while True:
        try:
            choice = input(prompt).strip().lower()
        except EOFError:
            choice = "c"

        if choice == "" or choice.startswith("s"):
            logger.info(
                "User requested to skip remaining circuits for this experiment."
            )
            return True

        if choice.startswith("c"):
            logger.info("User requested to cancel the experiment.")
            return False

        print("Please enter 's' to skip or 'c' to cancel.")


def _process_circuit(
    logger,
    evaluator,
    metric_ids: list[str],
    config: Config,
    benchmark: str,
    network_topology: str,
    num_qpus: int,
    partitioner_name: str,
    partitioner,
    circuit_id: str,
    circuit,
    run_prefix: str,
) -> CircuitProcessingResult:
    num_qubits = circuit.num_qubits
    partitioned_circuit = None

    # Create the network topology
    qubits_per_qpu = ((num_qubits + num_qpus - 1) // num_qpus) + 1
    network = build_network_topology(network_topology, num_qpus, qubits_per_qpu)

    # Decompose the circuit
    decomposed_circuit = circuit.decompose(reps=10)

    row = ExperimentResultRow.create(
        benchmark, network_topology, num_qpus, partitioner_name, circuit_id
    )
    timeout_qubits = None

    try:
        logger.debug(f"{run_prefix} Starting partitioning...")

        # Partition the circuit
        total_partitioning_started_time = perf_counter()
        outcome = partitioner.partition(
            decomposed_circuit,
            network,
            timeout=config.timeout_seconds,
        )
        total_partitioning_exec_time_s = (
            perf_counter() - total_partitioning_started_time
        )
        row.partitioning_metrics["total_partitioning_exec_time_s"] = (
            total_partitioning_exec_time_s
        )

        # Extract any partitioning metrics from the outcome (if available) and add to the row
        partitioning_metrics = dict(getattr(outcome, "metrics", {}) or {})
        row.partitioning_metrics.update(partitioning_metrics)

        if outcome.status != "success":
            # Handle partitioning timeout
            if outcome.status == "timeout":
                if outcome.metrics.get(PARTITIONER_DISTRIBUTION_COST_METRIC) is None:
                    row.mark_timeout(
                        outcome.reason
                        or f"Timeout after {config.timeout_seconds} seconds",
                        outcome.traceback or "",
                        config.timeout_seconds,
                    )
                    timeout_qubits = num_qubits
                    logger.warning(
                        f"{run_prefix} Partitioning timed out after {config.timeout_seconds} seconds"
                    )
                else:
                    row.mark_partial_success(
                        "Partitioning exceeded time limit but returned a result with distribution cost",
                        row.partitioning_metrics,
                    )
                    logger.warning(
                        f"{run_prefix} Partitioning exceeded time limit but returned a result with distribution cost"
                    )

            # Handle partitioning failure
            elif outcome.status == "failed":
                row.mark_failure(
                    _compact_error_message(outcome.reason, "partition failed"),
                    outcome.traceback or "",
                )
                logger.error(
                    f"{run_prefix} Partitioning failed: {outcome.reason}",
                )

            # Return early with no partitioned circuit
            return CircuitProcessingResult(
                row=row,
                partitioned_circuit=None,
                timeout_qubits=timeout_qubits,
            )

        partitioned_circuit = outcome.circuit

        # Evaluate the original and partitioned circuits
        evaluation_started_time = perf_counter()
        evaluation_results = evaluator.evaluate(
            [("input", decomposed_circuit), ("output", partitioned_circuit)],
            metrics=metric_ids,
        )
        total_evaluation_exec_time_s = perf_counter() - evaluation_started_time
        row.partitioning_metrics["total_evaluation_exec_time_s"] = (
            total_evaluation_exec_time_s
        )

        input_metrics = evaluation_results.get("input", {})
        output_metrics = evaluation_results.get("output", {})

        # Mark the row as successful and store all metrics
        row.mark_success(
            input_circuit_metrics={
                metric: input_metrics.get(metric) for metric in metric_ids
            },
            output_circuit_metrics={
                metric: output_metrics.get(metric) for metric in metric_ids
            },
            partitioning_metrics=row.partitioning_metrics,
        )

        logger.debug(f"{run_prefix} Partitioning completed")
        return CircuitProcessingResult(
            row=row,
            partitioned_circuit=partitioned_circuit,
            timeout_qubits=timeout_qubits,
        )

    # Handle keyboard interrupts
    except KeyboardInterrupt:
        if _handle_keyboard_interrupt(logger):
            row.mark_skipped("skipped by user")
            return CircuitProcessingResult(
                row=row,
                partitioned_circuit=None,
                skip_current_combo=True,
            )
        # Cancel the entire experiment
        raise SystemExit(0)

    # Handle any other exceptions during partitioning or evaluation
    except Exception as exc:
        error_msg = _compact_error_message(
            f"{type(exc).__name__}: {exc}", type(exc).__name__
        )
        tb = traceback.format_exc()
        logger.exception(f"{run_prefix} {error_msg}")
        row.mark_failure(error_msg, tb)
        return CircuitProcessingResult(row=row, partitioned_circuit=None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the experiment runner."""
    parser = argparse.ArgumentParser(description="Run the DQC evaluation experiment")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a YAML config file that overrides the bundled defaults.",
    )
    return parser.parse_args(argv)


def run_experiment(config: Config) -> int:
    logger = setup_logging(config.run_paths.run_dir)

    logger.info("=" * 80)
    logger.info("DQC Evaluation Experiment Started")
    logger.info("=" * 80)
    logger.info(f"Results directory: {config.run_paths.run_dir}")
    logger.info(f"Benchmarks: {config.benchmarks}")
    logger.info(f"Partitioners: {[p[0] for p in config.partitioners]}")
    logger.info(f"Network topologies: {config.network_topologies}")
    logger.info(f"Number of QPUs: {config.num_qpus}")
    timeout_skip_limit = max(config.skip_larger_circuits_after_timeout, 0)
    if timeout_skip_limit:
        logger.info(
            "Skip larger circuits after %s consecutive timeout(s)",
            timeout_skip_limit,
        )
    else:
        logger.info("Skip larger circuits after timeouts: disabled")
    logger.info(
        f"Built-in metrics: {config.metrics if config.metrics is not None else 'all'}"
    )
    logger.info(f"Custom metrics: {config.custom_metrics}")
    logger.info(f"Timeout: {config.timeout_seconds} seconds")
    logger.info("-" * 80)

    loader = CircuitLoader()
    evaluator = CircuitMetricsEvaluator()

    for metric_id in config.custom_metrics:
        if metric_id not in CUSTOM_METRIC_REGISTRY:
            available = ", ".join(sorted(CUSTOM_METRIC_REGISTRY))
            raise KeyError(
                f"Unknown custom metric '{metric_id}'. Available: {available}"
            )

        compute_fn, description = CUSTOM_METRIC_REGISTRY[metric_id]
        evaluator.register_metric(
            metric_id=metric_id,
            compute_fn=compute_fn,
            description=description,
            overwrite=True,
        )

    metric_ids = build_metric_ids(config.metrics, config.custom_metrics)

    recorder = ExperimentRecorder(config.run_paths, config.results_columns)
    recorder.write_run_metadata(config)
    failed_results = 0

    try:
        for benchmark in config.benchmarks:
            circuits = list(loader.load(benchmark))

            logger.info(f"[Benchmark: {benchmark} ({len(circuits)} circuits)]")

            for network_topology in config.network_topologies:
                logger.info(f"[Benchmark: {benchmark}, Topology: {network_topology}]")
                for num_qpus in config.num_qpus:
                    logger.info(
                        f"[Benchmark: {benchmark}, Topology: {network_topology}, QPUs: {num_qpus}]"
                    )
                    for partitioner_name, PartitionerClass in config.partitioners:
                        partitioner = PartitionerClass(
                            **config.partitioner_configs.get(partitioner_name, {})
                        )
                        skip_qubits_threshold = None
                        timeout_streak = 0
                        skip_current_combo = False

                        logger.info(
                            f"[Benchmark: {benchmark}, Topology: {network_topology}, QPUs: {num_qpus}, Partitioner: {partitioner_name}]"
                        )

                        for circuit_id, circuit in tqdm(
                            circuits,
                            desc=f"{benchmark} ({partitioner_name}, {network_topology}, {num_qpus} QPUs)",
                            total=len(circuits),
                            leave=True,
                        ):
                            run_prefix = f"[benchmark={benchmark} topology={network_topology} partitioner={partitioner_name} num_qpus={num_qpus} circuit={circuit_id}]"

                            # Handle skipping logic
                            skip_reason = None
                            num_qubits = circuit.num_qubits
                            if skip_current_combo:
                                skip_reason = "skipped by user interrupt for this benchmark/topology/partitioner/QPU combo"
                            elif (
                                config.skip_larger_circuits_after_timeout
                                and skip_qubits_threshold is not None
                                and num_qubits >= skip_qubits_threshold
                            ):
                                skip_reason = f"skipped due to earlier timeout ({config.timeout_seconds} seconds) at {skip_qubits_threshold} qubits"

                            if skip_reason is not None:
                                logger.debug(f"{run_prefix} Skipping: {skip_reason}")
                                row = ExperimentResultRow.create(
                                    benchmark,
                                    network_topology,
                                    num_qpus,
                                    partitioner_name,
                                    circuit_id,
                                )
                                row.mark_skipped(skip_reason)
                                recorder.record(row.to_dict(), partitioned_circuit=None)
                                continue

                            # Run the partitioning and evaluation for this circuit
                            result = _process_circuit(
                                logger,
                                evaluator,
                                metric_ids,
                                config,
                                benchmark,
                                network_topology,
                                num_qpus,
                                partitioner_name,
                                partitioner,
                                circuit_id,
                                circuit,
                                run_prefix,
                            )

                            recorder.record(
                                result.row.to_dict(),
                                partitioned_circuit=result.partitioned_circuit,
                            )

                            result_status = result.row.status["status"]
                            if result_status in {"failed", "partial_success"} or (
                                isinstance(result_status, str)
                                and result_status.startswith("timeout")
                            ):
                                failed_results += 1

                            if result.skip_current_combo:
                                skip_current_combo = True

                            if result.timeout_qubits is not None:
                                timeout_streak += 1
                                if (
                                    timeout_skip_limit > 0
                                    and timeout_streak >= timeout_skip_limit
                                ):
                                    skip_qubits_threshold = result.timeout_qubits
                            else:
                                timeout_streak = 0

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        raise
    except Exception:
        logger.exception("Critical error during experiment")
        raise
    finally:
        recorder.finalize(config)

        logger.info("=" * 80)
        logger.info("Experiment Completed - Summary Statistics")
        logger.info("=" * 80)

        recorder.print_summary()

        logger.info(f"Results saved to: {recorder.run_dir}")
        logger.info(f"Log file: {config.run_paths.run_dir / 'output.log'}")

    if failed_results:
        logger.error(
            "Experiment completed with %s failed or incomplete result(s)",
            failed_results,
        )
        return 1

    return 0


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    config = load_config(args.config)
    return run_experiment(config)


if __name__ == "__main__":
    # raise SystemExit(main())
    main()
