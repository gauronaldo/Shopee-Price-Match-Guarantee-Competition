"""Lightweight aggregate reports for classical retrieval benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _metric(result: dict[str, Any], split: str, name: str) -> float:
    return float(result[split]["retrieval"][name])


def render_report(results: dict[str, Any], figure_path: Path) -> str:
    """Render an aggregate Markdown report without exposing raw listing content."""
    metric_k = int(results["evaluation"]["average_precision_at"])
    rows = []
    for name in ("phash", "tfidf", "orb", "fusion", "pair_matcher"):
        baseline = results["baselines"][name]
        rows.append(
            "| {name} | {val_map:.4f} | {val_recall:.4f} | {threshold:.4f} | "
            "{test_map:.4f} | {test_recall:.4f} | {test_f1:.4f} | {seconds:.2f} |".format(
                name=name,
                val_map=_metric(baseline, "validation", f"map@{metric_k}"),
                val_recall=_metric(baseline, "validation", f"recall@{metric_k}"),
                threshold=float(baseline["selected_threshold"]),
                test_map=_metric(baseline, "test", f"map@{metric_k}"),
                test_recall=_metric(baseline, "test", f"recall@{metric_k}"),
                test_f1=float(baseline["test"]["pair"]["f1"]),
                seconds=float(baseline["runtime_seconds"]),
            )
        )
    fusion_weight = float(results["selection"]["fusion_text_weight"])
    fusion_delta = _metric(results["baselines"]["fusion"], "test", f"map@{metric_k}") - _metric(
        results["baselines"]["tfidf"], "test", f"map@{metric_k}"
    )
    pair_delta = float(results["baselines"]["pair_matcher"]["test"]["pair"]["f1"]) - float(
        results["baselines"]["fusion"]["test"]["pair"]["f1"]
    )
    efficiency = results["efficiency"]
    peak_working_set = efficiency["process_peak_working_set_bytes"]
    peak_memory_text = (
        f"{int(peak_working_set) / 2**20:.1f} MiB"
        if peak_working_set is not None
        else "unavailable"
    )
    provenance = results["provenance"]
    pair_model = results["model"]["pair_matcher"]
    candidate_ceiling = results["analysis"]["candidate_ceiling"]
    coefficients = results["model"]["pair_matcher"]["standardized_coefficients"]
    coefficient_rows = [
        f"| `{name}` | {float(value):+.4f} |"
        for name, value in sorted(
            coefficients.items(), key=lambda item: (-abs(float(item[1])), item[0])
        )
    ]
    failure_rows = [
        f"| `{name}` | {int(value):,} |"
        for name, value in sorted(
            results["analysis"]["pair_matcher_failure_counts"].items(),
            key=lambda item: (-int(item[1]), item[0]),
        )
    ]
    relative_figure = figure_path.as_posix()
    if relative_figure.startswith("reports/"):
        relative_figure = relative_figure.removeprefix("reports/")
    return "\n".join(
        [
            "# Classical retrieval benchmark",
            "",
            "This report contains aggregate validation/test results only. Retrieval uses the full",
            "corresponding split as its candidate pool and always excludes the query itself.",
            "TF-IDF vocabulary and IDF are fit on train only. Fusion weight and pair",
            "thresholds are selected on validation, then frozen for the final test evaluation.",
            "",
            "## Provenance",
            "",
            f"- Config: `{provenance['config_version']}` (`{provenance['config_sha256']}`)",
            f"- Split manifest SHA-256: `{provenance['manifest_sha256']}`",
            f"- Git commit / dirty: `{provenance['git_commit']}` / `{provenance['git_dirty']}`",
            f"- Seed: `{provenance['seed']}`",
            f"- Environment: Python `{provenance['python']}`, OpenCV `{provenance['opencv']}`, "
            f"NumPy `{provenance['numpy']}`, scikit-learn `{provenance['scikit_learn']}`",
            "",
            "## Results",
            "",
            f"Retrieval metrics are macro-averaged per query. Pair F1 is micro-averaged over "
            f"directed pairs and counts unretrieved positives as false negatives. Retrieval "
            f"columns use K={metric_k}.",
            "",
            "| Baseline | Val mAP | Val recall | Val threshold | Test mAP | Test recall | "
            "Test pair F1 | End-to-end runtime (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Selected fusion text weight: **{fusion_weight:.2f}**.",
            f"Fusion improves test mAP@{metric_k} over TF-IDF by **{fusion_delta:.4f}**.",
            "The pair matcher changes test pair F1 versus weighted fusion by "
            f"**{pair_delta:+.4f}**.",
            "Runtime covers validation plus test; ORB and pair matcher include their candidate",
            "stages. Pair-model training time is reported separately and excluded from latency.",
            "Pair-model fit data: {queries:,} sampled train queries, {pairs:,} directed pairs "
            "({positives:,} positive / {negatives:,} negative), {seconds:.2f} seconds.".format(
                queries=int(pair_model["training_queries"]),
                pairs=int(pair_model["training_pairs"]),
                positives=int(pair_model["training_positives"]),
                negatives=int(pair_model["training_negatives"]),
                seconds=float(pair_model["fit_seconds"]),
            ),
            "Mean end-to-end milliseconds/query: "
            + ", ".join(
                f"{name}={float(value):.2f}"
                for name, value in efficiency["mean_end_to_end_ms_per_query"].items()
            )
            + ".",
            f"Peak process working set: **{peak_memory_text}**.",
            "",
            f"![Validation threshold sweeps]({relative_figure})",
            "",
            "## Candidate ceiling",
            "",
            "The label-blind union contains the top candidates from pHash and TF-IDF before ORB",
            "or pair scoring. Its full-set recall is the maximum any downstream scorer can retain.",
            "",
            "| Split | Macro positive recall | Hit rate | Mean candidates/query | Max candidates |",
            "|---|---:|---:|---:|---:|",
            "| Validation | {macro_recall:.4f} | {hit_rate:.4f} | {mean_candidates:.2f} | "
            "{max_candidates:.0f} |".format(**candidate_ceiling["validation"]),
            "| Test | {macro_recall:.4f} | {hit_rate:.4f} | {mean_candidates:.2f} | "
            "{max_candidates:.0f} |".format(**candidate_ceiling["test"]),
            "",
            "## Pair matcher coefficients",
            "",
            "Coefficients operate on standardized features. Positive values support a match;",
            "negative values oppose one. Magnitude indicates influence within this fitted model,",
            "not causality.",
            "",
            "| Feature | Standardized coefficient |",
            "|---|---:|",
            *coefficient_rows,
            "",
            "## Structured pair-matcher failures",
            "",
            "Counts use directed test pairs at the validation-selected threshold. Bounded examples",
            "with local titles and IDs remain in the ignored artifact directory.",
            "",
            "| Failure category | Count |",
            "|---|---:|",
            *failure_rows,
            "",
            "## Sampled failure analysis",
            "",
            "Manual review of the ignored deterministic example file found semantically unrelated",
            "pHash neighbors, title matches that omit identity-critical model/variant details, and",
            "ORB matches driven by shared visual structure. Several high-scoring cross-label title",
            "pairs also look plausibly identical, consistent with the Phase 1 label-fragmentation",
            "warning. These cases remain evaluation errors; labels are not silently rewritten.",
            "",
            "## Interpretation guardrails",
            "",
            "- The supplied pHash is an image-appearance signal, not proof of product identity.",
            "- ORB reranks the label-blind union of pHash and TF-IDF candidates; its",
            "  retrieval ceiling is therefore limited by that candidate union.",
            "- Pair features are label-blind. Logistic Regression is fitted on train pairs only;",
            "  the decision threshold is selected on validation and frozen for test.",
            "- Test labels were used only after validation selected the fusion weight and",
            "  thresholds.",
            "- Local success/failure examples are saved under the ignored artifact directory for",
            "  manual review and are not redistributed.",
            "",
        ]
    )


def render_threshold_svg(curves: dict[str, list[dict[str, float]]]) -> str:
    """Render dependency-free validation precision/recall/F1 threshold curves."""
    width, height = 920, 620
    left, right, top, bottom = 75, 30, 45, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = {"precision": "#2563eb", "recall": "#dc2626", "f1": "#059669"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<title>Classical retrieval validation threshold sweeps</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="460" y="25" text-anchor="middle" font-family="sans-serif" '
        'font-size="18">Validation threshold sweeps</text>',
    ]
    names = tuple(curves)
    panel_width = plot_width / len(names)
    for panel_index, name in enumerate(names):
        x0 = left + panel_index * panel_width
        x1 = x0 + panel_width - 15
        y0, y1 = top, top + plot_height
        lines.extend(
            [
                f'<line x1="{x0:.1f}" y1="{y1}" x2="{x1:.1f}" y2="{y1}" stroke="#555"/>',
                f'<line x1="{x0:.1f}" y1="{y0}" x2="{x0:.1f}" y2="{y1}" stroke="#555"/>',
                f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 25}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="13">{name}</text>',
            ]
        )
        for metric, color in colors.items():
            points = " ".join(
                f"{x0 + point['threshold'] * (x1 - x0):.1f},{y1 - point[metric] * plot_height:.1f}"
                for point in curves[name]
            )
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
    for tick in range(6):
        value = tick / 5
        y = top + (1 - value) * plot_height
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.1f}</text>'
        )
    legend_x = width - 260
    for index, (metric, color) in enumerate(colors.items()):
        x = legend_x + index * 85
        lines.extend(
            [
                f'<line x1="{x}" y1="35" x2="{x + 18}" y2="35" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x + 23}" y="39" font-family="sans-serif" '
                f'font-size="11">{metric}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"
