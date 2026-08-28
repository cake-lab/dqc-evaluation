def ebit_count_metric(analysis):
    """Count EPR operations in the analyzed circuit."""
    ops = analysis.working_circuit.count_ops()
    return int(
        sum(count for op_name, count in ops.items() if str(op_name).upper() == "EPR")
    )


CUSTOM_METRIC_REGISTRY = {
    "ebit_count": (
        ebit_count_metric,
        "Number of EPR operations in the circuit.",
    ),
}
