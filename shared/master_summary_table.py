# shared/master_summary_table.py
from tabulate import tabulate

def print_master_summary(all_results):
    rows = []
    for category, images in all_results.items():
        if "error" in images:
            rows.append([category, "❌ Error", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue

        total = len(images)
        stats = {"tesseract":0, "easyocr":0, "paddleocr":0}
        avg_runtimes = []

        # Per-engine timeout/abort counters
        timeouts = {"tesseract":0, "easyocr":0, "paddleocr":0}
        aborted = {"tesseract":0, "easyocr":0, "paddleocr":0}

        for _, log in images.items():
            outputs = log.get("all_outputs", {})
            for engine in stats.keys():
                entry = outputs.get(engine, {})
                # Success detection
                if "text" in entry and entry.get("text", "").strip():
                    stats[engine] += 1
                # Runtime collection
                if "runtime" in entry and isinstance(entry["runtime"], (int, float)):
                    avg_runtimes.append(entry["runtime"])
                # Timeout / aborted counting
                if entry.get("timed_out"):
                    timeouts[engine] += 1
                if entry.get("aborted"):
                    aborted[engine] += 1

        avg_runtime = round(sum(avg_runtimes)/len(avg_runtimes), 3) if avg_runtimes else "-"

        rows.append([
            category,
            total,
            stats["tesseract"],
            stats["easyocr"],
            stats["paddleocr"],
            avg_runtime,
            f"T:{timeouts['tesseract']} | A:{aborted['tesseract']}",
            f"T:{timeouts['easyocr']} | A:{aborted['easyocr']}",
            f"T:{timeouts['paddleocr']} | A:{aborted['paddleocr']}",
        ])

    print("\n📊 Master OCR Summary Table")
    print(tabulate(
        rows,
        headers=[
            "Category","Images",
            "Tesseract","EasyOCR","PaddleOCR",
            "Avg Runtime",
            "Tesseract Status","EasyOCR Status","PaddleOCR Status"
        ]
    ))
