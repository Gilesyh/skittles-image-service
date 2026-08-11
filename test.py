import cv2
import numpy as np

# =====================================================
# SETTINGS
# =====================================================

IMAGE = "scorecard.jpg"

# =====================================================
# LOAD IMAGE
# =====================================================

image = cv2.imread(IMAGE)

if image is None:
    raise FileNotFoundError(f"Cannot open '{IMAGE}'")

display = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =====================================================
# EDGE DETECTION
# =====================================================

edges = cv2.Canny(gray, 50, 150)

# =====================================================
# LINE DETECTION
# =====================================================

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=80,
    minLineLength=150,
    maxLineGap=15
)

angles = []

accepted = 0
rejected = 0

# =====================================================
# PROCESS LINES
# =====================================================

if lines is None:
    raise Exception("No lines detected.")

print(f"Detected {len(lines)} line segments")

for line in lines:

    # OpenCV sometimes returns [[x1,y1,x2,y2]]
    # and sometimes [x1,y1,x2,y2]
    if len(line) == 1:
        x1, y1, x2, y2 = line[0]
    else:
        x1, y1, x2, y2 = line

    dx = x2 - x1
    dy = y2 - y1

    length = np.hypot(dx, dy)

    angle = np.degrees(np.arctan2(dy, dx))

    # Convert angle into range -90..90

    if angle > 90:
        angle -= 180

    if angle < -90:
        angle += 180

    # Keep only long-ish horizontal lines

    if abs(angle) < 20 and length > 150:

        accepted += 1
        angles.append(angle)

        cv2.line(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    else:

        rejected += 1

# =====================================================
# ESTIMATE SKEW
# =====================================================

if len(angles) == 0:
    raise Exception("No suitable horizontal lines found.")

angles = np.array(angles)

median_angle = float(np.median(angles))
mean_angle = float(np.mean(angles))
std = float(np.std(angles))

# =====================================================
# DRAW RESULT
# =====================================================

cv2.putText(
    display,
    f"Median angle: {median_angle:.2f} deg",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

# =====================================================
# SAVE IMAGE
# =====================================================

cv2.imwrite("detected_lines.jpg", display)

# =====================================================
# PRINT RESULTS
# =====================================================

print()
print("Image statistics")
print("----------------------------")
print(f"Accepted lines : {accepted}")
print(f"Rejected lines : {rejected}")
print(f"Median angle   : {median_angle:.2f}")
print(f"Mean angle     : {mean_angle:.2f}")
print(f"Std deviation  : {std:.2f}")
print()
print("Annotated image saved as detected_lines.jpg")