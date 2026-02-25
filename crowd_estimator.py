import cv2
import numpy as np
import argparse

# Window settings
WINDOW = "Crowd Estimator"
SMOOTH = 0.05          
DELAY = 30             

# Resize limits (auto-fit inside these bounds)
MAX_WIDTH = 1000
MAX_HEIGHT = 1500

# Crowd level labels and colors
LEVELS = ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]
COLORS = [(0,200,0), (0,200,200), (0,140,255), (0,0,220)]

# Density thresholds for level classification
THRS = [0.0220, 0.0235, 0.0245]

# Density-to-count interpolation range
D_MIN, D_MAX = 0.0175, 0.0262
COUNT_MIN, COUNT_MAX = -4, 30


# Determine crowd level from density
def get_level(d):
    if d < THRS[0]: return 0
    if d < THRS[1]: return 1
    if d < THRS[2]: return 2
    return 3


# Process frame → edge detection + density calculation
def process(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    density = np.count_nonzero(edges) / edges.size
    return edges, density


# Draw crowd info overlay
def draw_overlay(frame, density, count):
    lvl = get_level(density)
    color = COLORS[lvl]
    label = LEVELS[lvl]

    cv2.rectangle(frame, (0,0), (frame.shape[1],100), (15,15,15), -1)
    cv2.putText(frame, f"Density: {label}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Est. Count: ~{count}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,220,220), 2)


# Resize frame while keeping aspect ratio
def resize_frame(frame):
    h, w = frame.shape[:2]
    scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def main(src):
    # Open video/webcam source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Cannot open source")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1000, 700)

    smooth_d = None  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get edges and raw density
        edges, raw_d = process(frame)

        # Apply exponential smoothing
        smooth_d = raw_d if smooth_d is None else SMOOTH*raw_d + (1-SMOOTH)*smooth_d

        # Estimate crowd count from density
        count = int(np.interp(smooth_d, [D_MIN, D_MAX], [COUNT_MIN, COUNT_MAX]))

        # Convert edges to 3-channel image for overlay
        display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        draw_overlay(display, smooth_d, count)
        display = resize_frame(display)

        cv2.imshow(WINDOW, display)

        # Press 'q' to exit
        if cv2.waitKey(DELAY) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=0, help="Webcam index or video file path")
    args = ap.parse_args()

    try:
        source = int(args.src)
    except:
        source = args.src

    main(source)