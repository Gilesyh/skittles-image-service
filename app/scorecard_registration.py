import cv2
import numpy as np


def register_scorecard(image, template):
    """
    Locate the scorecard template within a photographed image,
    using a memory-conscious feature-matching approach.

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

    # ---------------------------------------------------------
    # 1. Downscale both images for feature detection
    # ---------------------------------------------------------

    max_dimension = 1400

    def resize_for_detection(img):
        height, width = img.shape[:2]
        largest = max(width, height)

        if largest <= max_dimension:
            return img.copy(), 1.0

        scale = max_dimension / largest

        resized = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        return resized, scale

    image_small, image_scale = resize_for_detection(image)
    template_small, template_scale = resize_for_detection(template)

    # ---------------------------------------------------------
    # 2. Convert to greyscale
    # ---------------------------------------------------------

    image_gray = cv2.cvtColor(
        image_small,
        cv2.COLOR_BGR2GRAY
    )

    template_gray = cv2.cvtColor(
        template_small,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------------------------------
    # 3. Detect a capped number of SIFT features
    # ---------------------------------------------------------

    sift = cv2.SIFT_create(
        nfeatures=1200
    )

    template_keypoints, template_descriptors = sift.detectAndCompute(
        template_gray,
        None
    )

    image_keypoints, image_descriptors = sift.detectAndCompute(
        image_gray,
        None
    )

    if template_descriptors is None:
        raise Exception(
            "Unable to find sufficient features in template."
        )

    if image_descriptors is None:
        raise Exception(
            "Unable to find sufficient features in photographed image."
        )

    # ---------------------------------------------------------
    # 4. Use FLANN rather than brute-force matching
    # ---------------------------------------------------------

    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    matcher = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    matches = matcher.knnMatch(
        template_descriptors,
        image_descriptors,
        k=2
    )

    # ---------------------------------------------------------
    # 5. Lowe ratio test
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 6. Build matched coordinate arrays
    # ---------------------------------------------------------

    template_points = np.float32([
        template_keypoints[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    image_points = np.float32([
        image_keypoints[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # ---------------------------------------------------------
    # 7. Estimate homography
    # ---------------------------------------------------------

    homography, mask = cv2.findHomography(
        template_points,
        image_points,
        cv2.RANSAC,
        5.0
    )

    if homography is None:
        raise Exception(
            "Unable to calculate scorecard homography."
        )

    # ---------------------------------------------------------
    # 8. Project template corners into photographed image
    # ---------------------------------------------------------

    template_height_small, template_width_small = template_gray.shape

    template_corners_small = np.float32([
        [0, 0],
        [template_width_small - 1, 0],
        [template_width_small - 1, template_height_small - 1],
        [0, template_height_small - 1]
    ]).reshape(-1, 1, 2)

    detected_corners_small = cv2.perspectiveTransform(
        template_corners_small,
        homography
    ).reshape(4, 2)

    # ---------------------------------------------------------
    # 9. Convert detected photograph coordinates back to
    #    original image dimensions
    # ---------------------------------------------------------

    detected_corners_original = (
        detected_corners_small / image_scale
    )

    image_height, image_width = image.shape[:2]

    # Keep coordinates within image bounds
    detected_corners_original[:, 0] = np.clip(
        detected_corners_original[:, 0],
        0,
        image_width - 1
    )

    detected_corners_original[:, 1] = np.clip(
        detected_corners_original[:, 1],
        0,
        image_height - 1
    )

    # ---------------------------------------------------------
    # 10. Diagnostics
    # ---------------------------------------------------------

    match_count = len(good_matches)

    inliers = (
        int(mask.sum())
        if mask is not None
        else 0
    )

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
            "top_left":
                detected_corners_original[0].tolist(),

            "top_right":
                detected_corners_original[1].tolist(),

            "bottom_right":
                detected_corners_original[2].tolist(),

            "bottom_left":
                detected_corners_original[3].tolist()
        },

        "image_width": int(image_width),
        "image_height": int(image_height),

        "image_detection_scale": float(image_scale),
        "template_detection_scale": float(template_scale),

        "homography": homography
    }