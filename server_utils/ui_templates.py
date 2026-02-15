def control_page(ocr_paused):
    """Generate HTML control page."""
    button_label = "Resume OCR" if ocr_paused else "Pause OCR"
    
    html = f"""
    <html>
    <head><title>OCR Live Runner</title></head>
    <body style="font-family:sans-serif; text-align:center;">
        <h1>OCR Live Runner</h1>
        <img src="/stream" width="640" height="480" style="border:1px solid #ccc;" />
        <div style="margin-top:20px;">
            <form action="/toggle" method="get" style="display:inline;">
                <button type="submit">{button_label}</button>
            </form>
            <form action="/quit" method="get" style="display:inline; margin-left:10px;">
                <button type="submit">Quit Server</button>
            </form>
        </div>
    </body>
    </html>
    """
    return html