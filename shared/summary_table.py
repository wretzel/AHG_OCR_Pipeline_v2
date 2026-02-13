# shared/summary_table.py

def print_ocr_summary(results_log):
    print("\n📊 **OCR Summary**")
    for engine in ["tesseract", "easyocr", "paddleocr"]:
        entry = results_log.get(engine, {})
        if "text" in entry:
            conf = f"{entry['confidence']:.2f}"
            corpus = f"{entry.get('corpus_score', 0.0):.2f}"
            rel = "Reliable" if entry["reliable"] else "Unreliable"
            time = f"{entry.get('runtime', 0.0):.3f} sec"

            print(f"- {engine.capitalize():<10}: Conf: {conf} | Corpus: {corpus} | {rel:<10} | {time}")
        elif entry.get("skipped"):
            print(f"- {engine.capitalize():<10}: ⏭️ Skipped")
        else:
            print(f"- {engine.capitalize():<10}: ❌ Failed")

    east_entry = results_log.get("east", {})
    if "regions" in east_entry:
        reliable_count = sum(1 for r in east_entry["regions"] if r.get("isReliable"))
        total = east_entry.get("region_count", len(east_entry["regions"]))
        time = f"{east_entry.get('runtime', 0.0):.3f} sec"
        print(f"- {'EAST':<10}: ✅ {reliable_count}/{total} reliable regions     | {time}")
    else:
        print(f"- {'EAST':<10}: ❌ Failed")

