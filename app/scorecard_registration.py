import cv2
import numpy as np


def register_scorecard(image, template):
    """
    Locate the scorecard template within a photographed image.

    Parameters
    ----------
    image : numpy.ndarray
        The photographed scorecard scene.

    template : numpy.ndarray
        Clean blank scorecard template.

    Returns
    -------
    dict
        Registration diagnostics and detected card corners.
    """

    # Convert both images to greyscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # SIFT finds distinctive visual features in both images.
    sift = cv2.SIFT_create()

    template_keypoints, template_descriptors = sift.detectAndCompute(
        template_gray,
        None
    )

    image_keypoints, image_descriptors = sift.detectAndCompute(
        image_gray,
        None
    )

    if template_descriptors is None or image_descriptors is None:
        raise Exception("Unable to find sufficient image features.")

    # Match template features against features in the photograph.
    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(
        template_descriptors,
        image_descriptors,
        k=2
    )

    # Lowe's ratio test removes ambiguous matches.
    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        first, second = pair

        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    if len(good_matches) < 10:
        raise Exception(
            f"Not enough reliable template matches: {len(good_matches)}"
        )

    # Coordinates of corresponding features.
    template_points = np.float32([
        template_keypoints[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    image_points = np.float32([
        image_keypoints[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # Calculate the perspective transformation between the clean
    # template and the photographed card.
    homography, mask = cv2.findHomography(
        template_points,
        image_points,
        cv2.RANSAC,
        5.0
    )

    if homography is None:
        raise Exception("Unable to calculate scorecard homography.")

    # Template dimensions
    template_height, template_width = template_gray.shape

    # Four corners of the clean template.
    template_corners = np.float32([
        [0, 0],
        [template_width - 1, 0],
        [template_width - 1, template_height - 1],
        [0, template_height - 1]
    ]).reshape(-1, 1, 2)

    # Project those corners into the photograph.
    detected_corners = cv2.perspectiveTransform(
        template_corners,
        homography
    ).reshape(4, 2)

    inliers = int(mask.sum()) if mask is not None else 0
    match_count = len(good_matches)

    inlier_ratio = (
        inliers / match_count
        if match_count > 0
        else 0.0
    )

    return {
        "matches": match_count,
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),

        "corners": {
            "top_left": detected_corners[0].tolist(),
            "top_right": detected_corners[1].tolist(),
            "bottom_right": detected_corners[2].tolist(),
            "bottom_left": detected_corners[3].tolist()
        },

        "template_width": int(template_width),
        "template_height": int(template_height),

        # Keep this internally for the next stage.
        # main.py will not return it directly as JSON.
        "homography": homography
    }