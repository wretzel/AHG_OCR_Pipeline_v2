# graphics/theme.py

# Font settings (OpenCV uses Hershey fonts by default)
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SUBTITLE = 0.8
FONT_SCALE_OCR = 0.6

THICKNESS_SUBTITLE = 2
THICKNESS_OCR = 1

# Colors (BGR)
COLOR_TEXT = (255, 255, 255)       # white
COLOR_BOX = (0, 0, 0)              # black
COLOR_OCR_TEXT = (255, 255, 255)
COLOR_OCR_BOX = (0, 0, 0)

# Alpha for box blending
BOX_ALPHA = 0.55

# Layout
PADDING = 10
LINE_SPACING = 6

MARGIN_BOTTOM = 40
MARGIN_TOP = 20
MAX_WIDTH_RATIO = 0.85  # max text width as fraction of frame width
