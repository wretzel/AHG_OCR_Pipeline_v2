from ocr_modules.pipeline_utils.pipeline import run_pipeline, print_pipeline_log

def ocr_task(cv_img, pil_img, models, executor, mode):
    """Run OCR pipeline and return text, confidence."""
    result = run_pipeline(cv_img, pil_img, models, executor=executor, mode=mode)
    text = result["final_result"].get("text", "")
    conf = result["final_result"].get("confidence", 0.0)
    print_pipeline_log(result)
    return text, conf