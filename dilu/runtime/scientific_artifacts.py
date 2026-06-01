import csv
import os
from io import StringIO
from typing import Any, Dict


def _join_reasons(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def build_metrics_table_csv(report: Dict[str, Any]) -> str:
    fieldnames = [
        "model",
        "primary_metric_name",
        "primary_metric_value",
        "primary_llm_metric_name",
        "primary_llm_metric_value",
        "secondary_joint_metric_name",
        "secondary_joint_metric_value",
        "primary_metric_valid",
        "scientific_validity_status",
        "episodes",
        "crash_rate",
        "no_collision_rate",
        "decision_timeout_rate_mean",
        "fallback_action_rate_mean",
        "response_strict_format_rate",
        "avg_ego_speed_mps",
        "task_completion_rate",
        "driving_score_behavior_v1",
        "driving_safety_score_v1",
        "driving_efficiency_score_v1",
        "driving_comfort_score_v1",
        "llm_driver_score_v1",
        "llm_output_contract_score_v1",
        "llm_runtime_reliability_score_v1",
        "llm_action_validity_score_v1",
        "llm_flow_recovery_independence_score_v1",
        "llm_safety_intervention_independence_score_v1",
        "llm_parser_independence_score_v1",
        "llm_intervention_independence_score_v1",
        "llm_latency_score_v1",
        "llm_resource_efficiency_score_v1",
        "dilu_joint_score_v1",
        "driving_score",
        "driving_score_v2",
        "primary_metric_invalid_reasons",
        "scientific_validity_reasons",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for aggregate in report.get("aggregates", []) or []:
        writer.writerow(
            {
                field: _join_reasons(aggregate.get(field))
                if field.endswith("_reasons")
                else aggregate.get(field)
                for field in fieldnames
            }
        )
    return buffer.getvalue()


def build_scientific_summary_markdown(report: Dict[str, Any]) -> str:
    spec = (report.get("metrics_config") or {}).get("primary_metric_spec") or {}
    lines = [
        "# Scientific Summary",
        "",
        f"Experiment: {report.get('experiment_id') or 'unknown'}",
        f"Primary metric policy: {spec.get('policy') or 'dilu_split_score_v1'}",
        f"Minimum claim episodes: {spec.get('minimum_claim_episodes') or 30}",
        "",
        "| Model | Driving primary | Driving value | LLM primary | LLM value | Joint value | Primary valid | Scientific status | Reasons |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for aggregate in report.get("aggregates", []) or []:
        lines.append(
            "| {model} | {driving_metric} | {driving_value} | {llm_metric} | {llm_value} | {joint_value} | {valid} | {status} | {reasons} |".format(
                model=aggregate.get("model"),
                driving_metric=aggregate.get("primary_driving_metric_name")
                or aggregate.get("primary_metric_name"),
                driving_value=aggregate.get("primary_driving_metric_value")
                if aggregate.get("primary_driving_metric_value") is not None
                else aggregate.get("primary_metric_value"),
                llm_metric=aggregate.get("primary_llm_metric_name"),
                llm_value=aggregate.get("primary_llm_metric_value"),
                joint_value=aggregate.get("secondary_joint_metric_value"),
                valid=aggregate.get("primary_metric_valid"),
                status=aggregate.get("scientific_validity_status"),
                reasons=_join_reasons(aggregate.get("scientific_validity_reasons")) or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_stats_appendix_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Stats Appendix",
        "",
        "Hypothesis tests are disabled in this reporting version because no baseline comparison is declared.",
        "",
    ]
    for aggregate in report.get("aggregates", []) or []:
        stats = aggregate.get("scientific_stats") or {}
        lines.extend(
            [
                f"## {aggregate.get('model')}",
                "",
                f"- n_episodes: {stats.get('n_episodes')}",
                f"- n_completed: {stats.get('n_completed')}",
                f"- n_runtime_valid: {stats.get('n_runtime_valid')}",
                f"- hypothesis_tests_enabled: {stats.get('hypothesis_tests_enabled')}",
                "",
                "| Metric | n | Mean | Std | Median | 95% CI | Warnings |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for metric, item in sorted((stats.get("continuous_metrics") or {}).items()):
            ci95 = item.get("ci95")
            ci_text = "" if ci95 is None else f"[{ci95[0]}, {ci95[1]}]"
            lines.append(
                "| {metric} | {n} | {mean} | {std} | {median} | {ci} | {warnings} |".format(
                    metric=metric,
                    n=item.get("n"),
                    mean=item.get("mean"),
                    std=item.get("std"),
                    median=item.get("median"),
                    ci=ci_text,
                    warnings=_join_reasons(item.get("warnings")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_scientific_analysis_artifacts(report: Dict[str, Any], analysis_dir: str) -> Dict[str, str]:
    os.makedirs(analysis_dir, exist_ok=True)
    outputs = {
        "scientific_summary_md": os.path.join(analysis_dir, "scientific_summary.md"),
        "stats_appendix_md": os.path.join(analysis_dir, "stats_appendix.md"),
        "metrics_table_csv": os.path.join(analysis_dir, "metrics_table.csv"),
    }
    contents = {
        outputs["scientific_summary_md"]: build_scientific_summary_markdown(report),
        outputs["stats_appendix_md"]: build_stats_appendix_markdown(report),
        outputs["metrics_table_csv"]: build_metrics_table_csv(report),
    }
    for path, content in contents.items():
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
    return outputs
