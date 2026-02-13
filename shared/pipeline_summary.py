# shared/pipeline_summary.py

import os
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def load_results(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def summarize_pipeline(results):
    engine_wins = Counter()
    confidence_scores = []
    reliability_flags = []
    runtime_totals = defaultdict(list)
    failures = []

    for category, images in results.items():
        for filename, result in images.items():
            if not isinstance(result, dict):
                continue

            winner = result.get("winner")
            conf = result.get("confidence", 0.0)
            reliable = result.get("reliable", False)
            runtime = result.get("runtime", 0.0)

            if winner:
                engine_wins[winner] += 1
                runtime_totals[winner].append(runtime)
            else:
                failures.append(filename)

            confidence_scores.append(conf)
            reliability_flags.append(reliable)

    return {
        "engine_wins": engine_wins,
        "confidence_scores": confidence_scores,
        "reliability_flags": reliability_flags,
        "runtime_totals": runtime_totals,
        "failures": failures
    }

def plot_summary(summary):
    # Engine win rates
    engines = list(summary["engine_wins"].keys())
    wins = list(summary["engine_wins"].values())
    plt.figure(figsize=(6, 4))
    plt.bar(engines, wins, color="skyblue")
    plt.title("Engine Win Rates")
    plt.ylabel("Wins")
    plt.xlabel("OCR Engine")
    plt.tight_layout()
    plt.show()

    # Confidence histogram
    plt.figure(figsize=(6, 4))
    plt.hist(summary["confidence_scores"], bins=10, color="lightgreen", edgecolor="black")
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Reliability breakdown
    reliable_count = sum(summary["reliability_flags"])
    total = len(summary["reliability_flags"])
    plt.figure(figsize=(4, 4))
    plt.pie([reliable_count, total - reliable_count], labels=["Reliable", "Unreliable"], autopct="%1.1f%%", colors=["green", "red"])
    plt.title("Reliability Breakdown")
    plt.tight_layout()
    plt.show()

    # Runtime per engine
    avg_runtimes = {
    engine: round(
        sum(t for t in times if t > 0) / len([t for t in times if t > 0]),
        2
        )
        for engine, times in summary["runtime_totals"].items()
        if any(t > 0 for t in times)
    }
    engines = list(avg_runtimes.keys())
    times = list(avg_runtimes.values())
    plt.figure(figsize=(6, 4))
    plt.bar(engines, times, color="orange")
    plt.title("Average Runtime per Engine")
    plt.ylabel("Seconds")
    plt.xlabel("OCR Engine")
    plt.tight_layout()
    plt.show()

def print_failure_cases(summary):
    if summary["failures"]:
        print("\n❌ Images with no reliable result:")
        for fname in summary["failures"]:
            print(f" - {fname}")
    else:
        print("\n✅ All images produced reliable results.")

def run_summary(json_path="testing/test_results/pipeline_outputs.json"):
    print(f"\n📊 Summarizing pipeline results from: {json_path}")
    results = load_results(json_path)
    summary = summarize_pipeline(results)
    plot_summary(summary)
    print_failure_cases(summary)

if __name__ == "__main__":
    run_summary()
