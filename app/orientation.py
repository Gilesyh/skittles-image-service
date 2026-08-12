import cv2
import numpy as np
import time


def analyse_orientation(image):
    """
    Estimate the rotation required to make the scorecard horizontal.

    Parameters
    ----------
    image : numpy.ndarray
        OpenCV image.

    Returns
    -------
    dict
        Orientation analysis.
    """

    # Create a copy that we'll draw our detected lines on.
    display = image.copy()

    # Convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # Detect line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=150,
        maxLineGap=15
    )

    if lines is None:
        raise Exception("No lines detected.")

    angles = []

    accepted = 0
    rejected = 0

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

        # Convert into range -90..90
        if angle > 90:
            angle -= 180

        if angle < -90:
            angle += 180

        # Keep only long-ish horizontal lines
        if abs(angle) < 20 and length > 150:

            accepted += 1
            angles.append(angle)

            # Draw accepted lines in green
            cv2.line(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        else:

            rejected += 1

    if len(angles) == 0:
        raise Exception("No suitable horizontal lines found.")

    angles = np.array(angles)

    median_angle = float(np.median(angles))
    mean_angle = float(np.mean(angles))
    std = float(np.std(angles))

    # Simple confidence estimate
    confidence = max(
        0.0,
        min(
            1.0,
            1.0 - (std / 10.0)
        )
    )

    # Save a debug image every run
    filename = f"debug_lines_{int(time.time())}.jpg"
    cv2.imwrite(filename, display)

    print()
    print("--------------------------------------")
    print("Orientation analysis")
    print("--------------------------------------")
    print(f"Debug image : {filename}")
    print(f"Accepted    : {accepted}")
    print(f"Rejected    : {rejected}")
    print(f"Median      : {median_angle:.2f}°")
    print(f"Mean        : {mean_angle:.2f}°")
    print(f"Std Dev     : {std:.2f}")
    print(f"Confidence  : {confidence:.2f}")
    print("--------------------------------------")
    print()

    return {
        "rotation_angle": median_angle,
        "confidence": confidence,
        "accepted_lines": accepted,
        "rejected_lines": rejected,
        "median_angle": median_angle,
        "mean_angle": mean_angle,
        "standard_deviation": std,
        "display": display
    }