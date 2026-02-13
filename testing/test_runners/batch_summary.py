# testing/test_runners/batch_summary.py

import os
import json

def summarize_engine(results):
    total_conf = 0
    total_corpus = 0
    total_reliable = 0
    total = 0
    total_runtime = 0
    failures = 0

    for entry in results:
        if "text" in entry:
            total += 1
            total_conf += entry["confidence"]
            total_corpus += entry.get("corpus_score", 0.0)
            total_reliable += 1 if entry["reliable"] else 0
            total_runtime += entry.get("runtime", 0)
        elif "error" in entry:
            failures += 1

    if total == 0:
        return {
            "avg_conf": "-", "avg_corpus": "-", "reliable_rate": "-",
            "avg_runtime": "-", "failures": failures
        }

    return {
        "avg_conf": round(total_conf / total, 2),
        "avg_corpus": round(total_corpus / total, 2),
        "reliable_rate": f"{round(100 * total_reliable / total)}%",
        "avg_runtime": f"{round(total_runtime / total, 3)} sec",
        "failures": failures
    }

def summarize_east(results):
    total_regions = 0
    reliable_regions = 0
    total_images = 0
    total_runtime = 0
    failures = 0

    for entry in results:
        if "regions" in entry:
            total_images += 1
            total_regions += len(entry["regions"])
            reliable_regions += sum(1 for r in entry["regions"] if r.get("isReliable"))
            total_runtime += entry.get("runtime", 0)
        elif "error" in entry:
            failures += 1

    if total_images == 0:
        return {
            "avg_regions": "-", "reliable_rate": "-", "avg_runtime": "-", "failures": failures
        }

    avg_regions = round(total_regions / total_images, 1)
    rate = f"{round(100 * reliable_regions / total_regions)}%" if total_regions else "0%"
    avg_runtime = f"{round(total_runtime / total_images, 3)} sec"

    return {
        "avg_regions": avg_regions,
        "reliable_rate": rate,
        "avg_runtime": avg_runtime,
        "failures": failures
    }

def summarize_batch(path="testing/test_results/batch_outputs.json"):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    print("\n📊 **Batch OCR Summary**\n")
    for category, images in data.items():
        print(f"📂 {category}")
        engine_logs = {"tesseract": [], "easyocr": [], "paddleocr": []}

        for result in images.values():
            for engine in engine_logs:
                if engine in result:
                    engine_logs[engine].append(result[engine])

        for engine in ["tesseract", "easyocr", "paddleocr"]:
            stats = summarize_engine(engine_logs[engine])
            print(f"- {engine.capitalize():<10}: "
                f"Avg Conf: {stats['avg_conf']} | "
                f"Corpus: {stats['avg_corpus']} | "
                f"Reliable: {stats['reliable_rate']} | "
                f"Runtime: {stats['avg_runtime']} | "
                f"Fails: {stats['failures']}")
        east_logs = [result["east"] for result in images.values() if "east" in result]
        east_stats = summarize_east(east_logs)

        print(f"- {'EAST':<10}: "
            f"Avg Regions: {east_stats['avg_regions']} | "
            f"Reliable: {east_stats['reliable_rate']} | "
            f"Runtime: {east_stats['avg_runtime']} | "
            f"Fails: {east_stats['failures']}")

        print()

if __name__ == "__main__":
    summarize_batch()
