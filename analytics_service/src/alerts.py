"""
Anomaly alerting feature: compares the most recent pipeline run (from
metrics_store.py) against the historical baseline for the same video source,
flags metrics that deviate significantly (simple z-score, no ML needed - the
"AI" part is entirely in explaining the anomaly, not detecting it), and asks
a local LLM (Ollama, self-hosted per the design doc - no cloud API) to write
a plain-English alert a non-technical store manager can act on.

Falls back to a templated sentence if Ollama isn't reachable, so the feature
degrades gracefully rather than hard-failing.

Usage:
    python alerts.py --video-source ../../data/raw/sample_video.mp4 [--z-threshold 1.5] [--model llama3]
"""
import argparse
import statistics

import requests

import metrics_store

MIN_BASELINE_SAMPLES = 3
OLLAMA_URL = "http://localhost:11434/api/generate"

METRICS = {
    "people_density": {
        "label": "average number of people visible per frame",
        "value_fn": lambda r: r["avg_people_per_frame"],
    },
    "visitor_rate": {
        "label": "unique visitors per 100 frames",
        "value_fn": lambda r: (r["unique_identities"] / r["frames_processed"] * 100)
        if r["frames_processed"] else None,
    },
    "gender_confidence": {
        "label": "average confidence of the gender prediction",
        "value_fn": lambda r: r["avg_gender_confidence"],
    },
    "age_confidence": {
        "label": "average confidence of the age prediction",
        "value_fn": lambda r: r["avg_age_confidence"],
    },
}


def compute_anomalies(runs, z_threshold=1.5):
    """runs: most-recent-first list of rows from metrics_store.get_runs().
    Returns a list of anomaly dicts for the newest run vs. the rest as baseline."""
    if len(runs) < MIN_BASELINE_SAMPLES + 1:
        return [], f"Need at least {MIN_BASELINE_SAMPLES + 1} logged runs for a baseline, have {len(runs)}."

    latest, baseline = runs[0], runs[1:]
    anomalies = []
    for key, spec in METRICS.items():
        latest_val = spec["value_fn"](latest)
        baseline_vals = [v for v in (spec["value_fn"](r) for r in baseline) if v is not None]
        if latest_val is None or len(baseline_vals) < MIN_BASELINE_SAMPLES:
            continue

        mean = statistics.mean(baseline_vals)
        stdev = statistics.stdev(baseline_vals) if len(baseline_vals) > 1 else 0
        if stdev == 0:
            continue

        z = (latest_val - mean) / stdev
        if abs(z) >= z_threshold:
            anomalies.append({
                "metric": key,
                "label": spec["label"],
                "value": round(latest_val, 3),
                "baseline_mean": round(mean, 3),
                "baseline_stdev": round(stdev, 3),
                "z_score": round(z, 2),
                "direction": "spike" if z > 0 else "drop",
                "baseline_n": len(baseline_vals),
            })
    return anomalies, None


def explain_with_llm(anomaly, model="llama3"):
    prompt = (
        "You are a monitoring assistant for a retail store video analytics system. "
        "Write ONE short, plain-English alert sentence (max 30 words) for a "
        "non-technical store manager describing this observation. Do not mention "
        "'z-score', 'standard deviation', or other statistics jargon. "
        "State only the facts given below - do not invent a cause (no guessing "
        "about sales, promotions, lighting, clothing, weather, or anything else "
        "not stated here). Do not restate the numbers in different units than "
        "given. Output only the sentence, nothing else.\n\n"
        f"Metric: {anomaly['label']}\n"
        f"Current value: {anomaly['value']}\n"
        f"Typical value (from {anomaly['baseline_n']} prior observations): {anomaly['baseline_mean']}\n"
        f"Direction: this is a {anomaly['direction']} versus the typical value.\n"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        return _fallback_explanation(anomaly, reason=str(e))


def _fallback_explanation(anomaly, reason=None):
    direction_word = "well above" if anomaly["z_score"] > 0 else "well below"
    note = f" (LLM unavailable: {reason})" if reason else ""
    return (
        f"Alert: {anomaly['label']} was {anomaly['value']}, {direction_word} "
        f"the typical {anomaly['baseline_mean']} seen across the last "
        f"{anomaly['baseline_n']} runs.{note}"
    )


def run_alerts(video_source, z_threshold=1.5, model="llama3"):
    runs = metrics_store.get_runs(video_source=video_source)
    anomalies, skip_reason = compute_anomalies(runs, z_threshold)

    if skip_reason:
        print(skip_reason)
        return []

    if not anomalies:
        print(f"No anomalies detected (checked {len(METRICS)} metrics against "
              f"{len(runs) - 1} prior runs, threshold={z_threshold}).")
        return []

    print(f"{len(anomalies)} anomaly(ies) detected:\n")
    messages = []
    for a in anomalies:
        message = explain_with_llm(a, model=model)
        messages.append(message)
        print(f"[{a['metric']}] z={a['z_score']} -> {message}\n")
    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-source", required=True)
    parser.add_argument("--z-threshold", type=float, default=1.5)
    parser.add_argument("--model", default="llama3")
    args = parser.parse_args()

    run_alerts(args.video_source, args.z_threshold, args.model)
