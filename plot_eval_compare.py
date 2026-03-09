import argparse
import json
import os
import math

import matplotlib.pyplot as plt

from dilu.runtime import build_model_root, ensure_dir, read_json, write_json_atomic


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_value(v, default=0.0):
    return default if v is None else v


def _normalize_aggregates(report: dict):
    """Support both eval reports ('aggregates': [...]) and run reports ('aggregate': {...})."""
    aggregates = report.get("aggregates")
    if isinstance(aggregates, list) and aggregates:
        return aggregates, "eval"

    single = report.get("aggregate")
    if isinstance(single, dict) and single:
        model_name = report.get("chat_model") or report.get("model") or "run_model"
        normalized = dict(single)
        normalized["model"] = model_name
        return [normalized], "run"

    raise ValueError("No 'aggregates' list or 'aggregate' object found in report.")


def _plot_grid(models, charts, title: str, output_path: str) -> None:
    n = len(charts)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if isinstance(axes, plt.Axes):
        flat_axes = [axes]
    else:
        flat_axes = []
        for row_axes in (axes if isinstance(axes, (list, tuple)) else axes.tolist()):
            if isinstance(row_axes, (list, tuple)):
                flat_axes.extend(list(row_axes))
            else:
                flat_axes.append(row_axes)

    for ax, chart in zip(flat_axes[:n], charts):
        values = chart["values"]
        bars = ax.bar(models, values, color=chart["color"], alpha=0.9)
        ax.set_title(chart["title"])
        ax.set_xticklabels(models, rotation=20, ha="right")
        if chart["ylim"] is not None:
            ax.set_ylim(*chart["ylim"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if isinstance(value, float) else str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Hide any unused axes when chart count is odd.
    for ax in flat_axes[n:]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_aggregates(report: dict, output_path: str, extended: bool = False, all_metrics: bool = False) -> None:
    aggregates, source_type = _normalize_aggregates(report)

    models = [row["model"] for row in aggregates]
    title_prefix = "DiLu Run Metrics" if source_type == "run" else "DiLu Model Comparison"

    if all_metrics:
        charts = [
            {"values": [_safe_value(row.get("crash_rate")) for row in aggregates], "title": "Crash Rate", "ylim": (0, 1), "color": "#d95f02"},
            {"values": [_safe_value(row.get("no_collision_rate")) for row in aggregates], "title": "No-Collision Rate", "ylim": (0, 1), "color": "#1b9e77"},
            {"values": [_safe_value(row.get("avg_steps")) for row in aggregates], "title": "Average Steps", "ylim": None, "color": "#7570b3"},
            {"values": [_safe_value(row.get("avg_episode_runtime_sec")) for row in aggregates], "title": "Avg Episode Runtime (s)", "ylim": None, "color": "#66a61e"},
            {"values": [_safe_value(row.get("ttc_danger_rate_mean")) for row in aggregates], "title": "TTC Danger Rate", "ylim": (0, 1), "color": "#e41a1c"},
            {"values": [_safe_value(row.get("headway_violation_rate_mean")) for row in aggregates], "title": "Headway Violation Rate", "ylim": (0, 1), "color": "#ff7f00"},
            {"values": [_safe_value(row.get("lane_change_rate_mean")) for row in aggregates], "title": "Lane Change Rate", "ylim": (0, 1), "color": "#377eb8"},
            {"values": [_safe_value(row.get("flap_accel_decel_rate_mean")) for row in aggregates], "title": "Accel/Decel Flap Rate", "ylim": (0, 1), "color": "#984ea3"},
            {"values": [_safe_value(row.get("avg_reward_sum")) for row in aggregates], "title": "Average Reward Sum", "ylim": None, "color": "#4daf4a"},
            {"values": [_safe_value(row.get("avg_reward_per_step")) for row in aggregates], "title": "Average Reward/Step", "ylim": None, "color": "#a65628"},
            {"values": [_safe_value(row.get("avg_ego_speed_mps")) for row in aggregates], "title": "Average Ego Speed (m/s)", "ylim": None, "color": "#f781bf"},
            {"values": [_safe_value(row.get("format_failure_rate_mean")) for row in aggregates], "title": "Format Failure Rate", "ylim": (0, 1), "color": "#999999"},
            {"values": [_safe_value(row.get("decision_latency_ms_avg")) for row in aggregates], "title": "Decision Latency (ms)", "ylim": None, "color": "#17becf"},
        ]
        _plot_grid(models, charts, f"{title_prefix} (All Metrics)", output_path)
        return

    if not extended:
        charts = [
            {"values": [_safe_value(row.get("crash_rate")) for row in aggregates], "title": "Crash Rate", "ylim": (0, 1), "color": "#d95f02"},
            {"values": [_safe_value(row.get("no_collision_rate")) for row in aggregates], "title": "No-Collision Rate", "ylim": (0, 1), "color": "#1b9e77"},
            {"values": [_safe_value(row.get("avg_steps")) for row in aggregates], "title": "Average Steps", "ylim": None, "color": "#7570b3"},
            {"values": [_safe_value(row.get("avg_episode_runtime_sec")) for row in aggregates], "title": "Avg Episode Runtime (s)", "ylim": None, "color": "#66a61e"},
        ]
        _plot_grid(models, charts, title_prefix, output_path)
        return

    charts = [
        {"values": [_safe_value(row.get("ttc_danger_rate_mean")) for row in aggregates], "title": "TTC Danger Rate", "ylim": (0, 1), "color": "#e41a1c"},
        {"values": [_safe_value(row.get("headway_violation_rate_mean")) for row in aggregates], "title": "Headway Violation Rate", "ylim": (0, 1), "color": "#ff7f00"},
        {"values": [_safe_value(row.get("lane_change_rate_mean")) for row in aggregates], "title": "Lane Change Rate", "ylim": (0, 1), "color": "#377eb8"},
        {"values": [_safe_value(row.get("flap_accel_decel_rate_mean")) for row in aggregates], "title": "Accel/Decel Flap Rate", "ylim": (0, 1), "color": "#984ea3"},
    ]
    _plot_grid(models, charts, f"{title_prefix} (Extended Safety/Comfort)", output_path)


def emit_per_model_plots(report: dict, all_metrics: bool, extended: bool) -> list:
    """Emit one chart per model under each model's plots folder (for eval reports)."""
    aggregates, source_type = _normalize_aggregates(report)
    if source_type != "eval":
        return []

    experiment_root = report.get("experiment_root")
    if not experiment_root:
        return []

    outputs = []
    for row in aggregates:
        model_name = row.get("model", "model")
        model_root = build_model_root(experiment_root, model_name)
        plots_dir = ensure_dir(os.path.join(model_root, "plots"))
        suffix = "all" if all_metrics else ("extended" if extended else "default")
        output_path = os.path.join(plots_dir, f"model_metrics_{suffix}.png")
        single_report = {
            "aggregate": row,
            "chat_model": model_name,
        }
        plot_aggregates(single_report, output_path, extended=extended, all_metrics=all_metrics)
        outputs.append(output_path)
    return outputs


def update_manifest_for_plots(report: dict, global_plot_path: str, per_model_plot_paths: list) -> None:
    experiment_root = report.get("experiment_root")
    if not experiment_root:
        return

    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})
    compare_meta = manifest.setdefault("compare", {})
    compare_meta["latest_plot"] = global_plot_path

    plot_history = compare_meta.setdefault("plot_history", [])
    if global_plot_path not in plot_history:
        plot_history.append(global_plot_path)

    models = manifest.setdefault("models", {})
    for path in per_model_plot_paths:
        parts = path.replace("\\", "/").split("/")
        if "models" not in parts:
            continue
        idx = parts.index("models")
        if idx + 1 >= len(parts):
            continue
        model_slug = parts[idx + 1]
        for model_name, model_meta in models.items():
            if str(model_meta.get("slug")) == model_slug:
                model_meta["latest_plot"] = path
                break

    write_json_atomic(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregate metrics from evaluate_models_ollama.py JSON output.")
    parser.add_argument("-i", "--input", required=True, help="Path to comparison JSON report")
    parser.add_argument("-o", "--output", default=None, help="Output image path (PNG). Defaults next to input file.")
    parser.add_argument("--extended", action="store_true", help="Plot extended metrics (TTC/headway/lane-change/flapping).")
    parser.add_argument("--all-metrics", action="store_true", help="Plot all available aggregate metrics in one figure.")
    parser.add_argument("--emit-per-model", action="store_true", help="Emit one plot per model under experiment model plot folders.")
    args = parser.parse_args()

    report = load_report(args.input)

    if args.output:
        output_path = args.output
    else:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_plot.png"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_aggregates(report, output_path, extended=args.extended, all_metrics=args.all_metrics)
    print(f"Saved plot: {output_path}")

    per_model_outputs = []
    if args.emit_per_model:
        per_model_outputs = emit_per_model_plots(report, all_metrics=args.all_metrics, extended=args.extended)
        if per_model_outputs:
            print("Saved per-model plots:")
            for path in per_model_outputs:
                print(f"- {path}")
        else:
            print("No per-model plots emitted (report is not an experiment eval report with experiment_root).")

    update_manifest_for_plots(report, output_path, per_model_outputs)


if __name__ == "__main__":
    main()
