# shared/text_output.py

def print_ocr_text_outputs(results_log):
    print("📝 **OCR Outputs**")
    for engine in ["tesseract", "easyocr", "paddleocr"]:
        entry = results_log.get(engine, {})
        print(f"\n🧠 {engine.capitalize()}")
        if "text" in entry:
            text = entry["text"].strip()
            print(text if text else "[No text detected]")
        elif entry.get("skipped"):
            print("[Engine skipped]")
        else:
            print("[Engine failed]")

    east_entry = results_log.get("east", {})
    print("\n🧠 EAST (Text Regions)")
    if "regions" in east_entry:
        regions = sorted(east_entry["regions"], key=lambda r: r["confidence"], reverse=True)
        total = len(regions)
        preview = regions[:5]

        for i, region in enumerate(preview):
            box = region["box"]
            conf = region["confidence"]
            rel = "✅" if region["isReliable"] else "⚠️"
            print(f"  Region {i+1}: {rel} {conf:.2f} @ {box}")

        if total > 5:
            print(f"  ...and {total - 5} more regions not shown")
    else:
        print("[EAST failed]")
