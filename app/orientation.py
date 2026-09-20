import cv2
import numpy as np
import time


def analyse_orientation(image):
    """
    Analyse a scorecard image for:

    1. Horizontal-line orientation, used to determine the deskew angle.
    2. Significant near-vertical line structures, which may later be used
       by the YSSL workflow to identify the Home/Away dividing rule.

    IMPORTANT:
    ----------
    The vertical-line analysis performed here is diagnostic/geometric only.

    It does NOT assume that the image is a YSSL card.
    It does NOT select a Home/Away split.
    It does NOT affect the calculated rotation angle.

    This means YSL cards can continue to use this endpoint solely for
    orientation, while YSSL processing can make use of the additional
    vertical-line information downstream.

    Parameters
    ----------
    image : numpy.ndarray
        OpenCV BGR image.

    Returns
    -------
    dict
        Orientation analysis plus vertical-line diagnostics.
    """

    # =========================================================
    # BASIC IMAGE INFORMATION
    # =========================================================

    image_height, image_width = image.shape[:2]

    if image_width <= 0 or image_height <= 0:
        raise Exception("Image has invalid dimensions.")

    display = image.copy()

    # =========================================================
    # GREYSCALE + EDGE DETECTION
    # =========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # =========================================================
    # HOUGH LINE DETECTION
    # =========================================================

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

    # =========================================================
    # HORIZONTAL ORIENTATION ANALYSIS
    # =========================================================

    horizontal_angles = []

    horizontal_accepted = 0
    horizontal_rejected = 0

    # =========================================================
    # VERTICAL SEGMENT COLLECTION
    # =========================================================

    vertical_segments = []

    for line in lines:

        # OpenCV can return either:
        #
        # [[x1, y1, x2, y2]]
        #
        # or:
        #
        # [x1, y1, x2, y2]

        if len(line) == 1:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        dx = x2 - x1
        dy = y2 - y1

        length = float(
            np.hypot(dx, dy)
        )

        if length <= 0:
            continue

        angle = float(
            np.degrees(
                np.arctan2(dy, dx)
            )
        )

        # Convert line angle into -90..90.
        if angle > 90:
            angle -= 180

        if angle < -90:
            angle += 180

        # -----------------------------------------------------
        # HORIZONTAL LINES
        # -----------------------------------------------------

        if (
            abs(angle) < 20
            and length > 150
        ):

            horizontal_accepted += 1

            horizontal_angles.append(
                angle
            )

            # Green diagnostic line.
            cv2.line(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        else:
            horizontal_rejected += 1

        # -----------------------------------------------------
        # NEAR-VERTICAL LINES
        #
        # A perfect vertical line has an angle of +/-90 degrees.
        #
        # We deliberately keep this reasonably strict because
        # handwriting and card edges can otherwise create many
        # false candidates.
        # -----------------------------------------------------

        vertical_deviation = abs(
            90.0 - abs(angle)
        )

        minimum_vertical_length = max(
            100.0,
            image_height * 0.12
        )

        if (
            vertical_deviation <= 8.0
            and length >= minimum_vertical_length
        ):

            # Put top endpoint first.
            if y1 <= y2:
                top_x = x1
                top_y = y1
                bottom_x = x2
                bottom_y = y2
            else:
                top_x = x2
                top_y = y2
                bottom_x = x1
                bottom_y = y1

            midpoint_x = (
                top_x + bottom_x
            ) / 2.0

            vertical_segments.append({
                "midpoint_x":
                    float(midpoint_x),

                "top_x":
                    int(top_x),

                "top_y":
                    int(top_y),

                "bottom_x":
                    int(bottom_x),

                "bottom_y":
                    int(bottom_y),

                "length":
                    float(length),

                "angle":
                    float(angle),

                "vertical_deviation":
                    float(vertical_deviation)
            })

            # Blue diagnostic line.
            cv2.line(
                display,
                (top_x, top_y),
                (bottom_x, bottom_y),
                (255, 0, 0),
                2
            )

    # =========================================================
    # CALCULATE ORIENTATION
    # =========================================================

    if len(horizontal_angles) == 0:
        raise Exception(
            "No suitable horizontal lines found."
        )

    horizontal_angles_array = np.array(
        horizontal_angles,
        dtype=np.float32
    )

    median_angle = float(
        np.median(
            horizontal_angles_array
        )
    )

    mean_angle = float(
        np.mean(
            horizontal_angles_array
        )
    )

    standard_deviation = float(
        np.std(
            horizontal_angles_array
        )
    )

    confidence = max(
        0.0,
        min(
            1.0,
            1.0 - (
                standard_deviation / 10.0
            )
        )
    )

    # =========================================================
    # CLUSTER VERTICAL SEGMENTS BY X POSITION
    # =========================================================
    #
    # HoughLinesP commonly detects both edges of the same
    # printed vertical rule and/or detects one rule as several
    # separate line fragments.
    #
    # Therefore individual segments should NOT be treated as
    # individual card rules.
    #
    # We cluster nearby X positions first.
    # =========================================================

    cluster_tolerance = max(
        6.0,
        image_width * 0.008
    )

    vertical_segments_sorted = sorted(
        vertical_segments,
        key=lambda segment:
            segment["midpoint_x"]
    )

    raw_clusters = []

    for segment in vertical_segments_sorted:

        best_cluster = None
        best_distance = None

        for cluster in raw_clusters:

            cluster_x = float(
                np.average(
                    [
                        item["midpoint_x"]
                        for item
                        in cluster
                    ],
                    weights=[
                        item["length"]
                        for item
                        in cluster
                    ]
                )
            )

            distance = abs(
                segment["midpoint_x"]
                - cluster_x
            )

            if (
                distance <= cluster_tolerance
                and (
                    best_distance is None
                    or distance < best_distance
                )
            ):
                best_cluster = cluster
                best_distance = distance

        if best_cluster is None:
            raw_clusters.append(
                [segment]
            )
        else:
            best_cluster.append(
                segment
            )

    # =========================================================
    # SUMMARISE EACH VERTICAL CLUSTER
    # =========================================================

    vertical_clusters = []

    for cluster in raw_clusters:

        lengths = np.array(
            [
                item["length"]
                for item
                in cluster
            ],
            dtype=np.float64
        )

        midpoints = np.array(
            [
                item["midpoint_x"]
                for item
                in cluster
            ],
            dtype=np.float64
        )

        if lengths.sum() > 0:
            cluster_x = float(
                np.average(
                    midpoints,
                    weights=lengths
                )
            )
        else:
            cluster_x = float(
                np.mean(midpoints)
            )

        top_item = min(
            cluster,
            key=lambda item:
                item["top_y"]
        )

        bottom_item = max(
            cluster,
            key=lambda item:
                item["bottom_y"]
        )

        top_y = int(
            min(
                item["top_y"]
                for item
                in cluster
            )
        )

        bottom_y = int(
            max(
                item["bottom_y"]
                for item
                in cluster
            )
        )

        vertical_span = max(
            0,
            bottom_y - top_y
        )

        longest_segment = float(
            max(
                item["length"]
                for item
                in cluster
            )
        )

        total_detected_length = float(
            sum(
                item["length"]
                for item
                in cluster
            )
        )

        x_fraction = (
            cluster_x / image_width
        )

        vertical_span_fraction = (
            vertical_span / image_height
        )

        longest_segment_fraction = (
            longest_segment / image_height
        )

        # This is deliberately only a broad diagnostic corridor.
        #
        # It MUST NOT be interpreted as:
        # "this is the Home/Away divider".
        #
        # The scorecard itself may not be centred in the photo.
        in_central_corridor = (
            0.30 <= x_fraction <= 0.70
        )

        vertical_clusters.append({
            "x":
                float(cluster_x),

            "x_fraction":
                float(x_fraction),

            "representative_top_x":
                int(top_item["top_x"]),

            "representative_bottom_x":
                int(bottom_item["bottom_x"]),

            "top_y":
                int(top_y),

            "bottom_y":
                int(bottom_y),

            "vertical_span":
                int(vertical_span),

            "vertical_span_fraction":
                float(vertical_span_fraction),

            "longest_segment":
                float(longest_segment),

            "longest_segment_fraction":
                float(longest_segment_fraction),

            "segment_count":
                int(len(cluster)),

            "total_detected_length":
                float(total_detected_length),

            "in_central_corridor":
                bool(in_central_corridor)
        })

    # =========================================================
    # SORT CLUSTERS LEFT -> RIGHT
    # =========================================================
    #
    # The previous output happened to be ordered largely by
    # strength. For structural analysis, explicit left-to-right
    # ordering is much more useful.
    # =========================================================

    vertical_clusters.sort(
        key=lambda cluster:
            cluster["x"]
    )

    # =========================================================
    # ADD LEFT-TO-RIGHT INDEX
    # =========================================================

    for index, cluster in enumerate(
        vertical_clusters
    ):
        cluster["left_to_right_index"] = int(
            index
        )

    # =========================================================
    # ANALYSE GAPS BETWEEN VERTICAL RULES
    # =========================================================
    #
    # A YSSL card contains a structured collection of vertical
    # rules. Their RELATIVE spacing is useful information.
    #
    # We therefore return the gaps between detected clusters,
    # but we still do not decide that any particular gap/rule
    # represents the Home/Away boundary.
    # =========================================================

    vertical_cluster_gaps = []

    for index in range(
        len(vertical_clusters) - 1
    ):

        left_cluster = (
            vertical_clusters[index]
        )

        right_cluster = (
            vertical_clusters[index + 1]
        )

        gap = (
            right_cluster["x"]
            - left_cluster["x"]
        )

        vertical_cluster_gaps.append({
            "left_cluster_index":
                int(index),

            "right_cluster_index":
                int(index + 1),

            "left_x":
                float(
                    left_cluster["x"]
                ),

            "right_x":
                float(
                    right_cluster["x"]
                ),

            "gap":
                float(gap),

            "gap_fraction":
                float(
                    gap / image_width
                )
        })

    # =========================================================
    # BUILD POSSIBLE STRUCTURAL PAIRS
    # =========================================================
    #
    # For every pair of significant vertical clusters we return:
    #
    # - their positions
    # - separation
    # - midpoint
    # - overlap in Y
    #
    # This gives downstream YSSL logic enough geometry to combine
    # with OCR landmarks without having to repeat OpenCV work.
    # =========================================================

    vertical_cluster_pairs = []

    for left_index in range(
        len(vertical_clusters)
    ):

        for right_index in range(
            left_index + 1,
            len(vertical_clusters)
        ):

            left_cluster = (
                vertical_clusters[left_index]
            )

            right_cluster = (
                vertical_clusters[right_index]
            )

            separation = (
                right_cluster["x"]
                - left_cluster["x"]
            )

            midpoint = (
                left_cluster["x"]
                + right_cluster["x"]
            ) / 2.0

            overlap_top = max(
                left_cluster["top_y"],
                right_cluster["top_y"]
            )

            overlap_bottom = min(
                left_cluster["bottom_y"],
                right_cluster["bottom_y"]
            )

            overlap_span = max(
                0,
                overlap_bottom - overlap_top
            )

            overlap_fraction = (
                overlap_span / image_height
            )

            vertical_cluster_pairs.append({
                "left_cluster_index":
                    int(left_index),

                "right_cluster_index":
                    int(right_index),

                "left_x":
                    float(
                        left_cluster["x"]
                    ),

                "right_x":
                    float(
                        right_cluster["x"]
                    ),

                "separation":
                    float(separation),

                "separation_fraction":
                    float(
                        separation / image_width
                    ),

                "midpoint_x":
                    float(midpoint),

                "midpoint_x_fraction":
                    float(
                        midpoint / image_width
                    ),

                "vertical_overlap":
                    int(overlap_span),

                "vertical_overlap_fraction":
                    float(overlap_fraction)
            })

    # =========================================================
    # IDENTIFY GEOMETRICALLY STRONG CLUSTERS
    # =========================================================
    #
    # These are NOT claimed to be Home/Away dividers.
    #
    # They are simply clusters with enough vertical presence to
    # be useful candidates downstream.
    # =========================================================

    strong_vertical_cluster_indices = []

    for index, cluster in enumerate(
        vertical_clusters
    ):

        strong_enough = (
            cluster["vertical_span_fraction"]
            >= 0.20
            or
            cluster["longest_segment_fraction"]
            >= 0.20
        )

        if strong_enough:
            strong_vertical_cluster_indices.append(
                int(index)
            )

    # =========================================================
    # SAVE DEBUG IMAGE
    # =========================================================

    filename = (
        f"debug_lines_{int(time.time())}.jpg"
    )

    cv2.imwrite(
        filename,
        display
    )

    # =========================================================
    # SERVER LOGGING
    # =========================================================

    print()
    print(
        "--------------------------------------"
    )
    print(
        "Orientation analysis"
    )
    print(
        "--------------------------------------"
    )

    print(
        f"Debug image : {filename}"
    )

    print(
        f"Image size  : "
        f"{image_width} x {image_height}"
    )

    print(
        f"Accepted H  : "
        f"{horizontal_accepted}"
    )

    print(
        f"Rejected H  : "
        f"{horizontal_rejected}"
    )

    print(
        f"Median      : "
        f"{median_angle:.2f}°"
    )

    print(
        f"Mean        : "
        f"{mean_angle:.2f}°"
    )

    print(
        f"Std Dev     : "
        f"{standard_deviation:.2f}"
    )

    print(
        f"Confidence  : "
        f"{confidence:.2f}"
    )

    print(
        f"Vertical seg: "
        f"{len(vertical_segments)}"
    )

    print(
        f"Vertical cls: "
        f"{len(vertical_clusters)}"
    )

    print(
        "Vertical clusters:"
    )

    for index, cluster in enumerate(
        vertical_clusters
    ):
        print(
            f"  {index}: "
            f"x={cluster['x']:.1f} "
            f"({cluster['x_fraction']:.3f}), "
            f"span={cluster['vertical_span_fraction']:.3f}, "
            f"segments={cluster['segment_count']}"
        )

    print(
        "--------------------------------------"
    )
    print()

    # =========================================================
    # RETURN
    # =========================================================

    return {
        # -----------------------------------------------------
        # EXISTING ORIENTATION OUTPUT
        # -----------------------------------------------------

        "rotation_angle":
            float(median_angle),

        "confidence":
            float(confidence),

        "accepted_lines":
            int(horizontal_accepted),

        "rejected_lines":
            int(horizontal_rejected),

        "median_angle":
            float(median_angle),

        "mean_angle":
            float(mean_angle),

        "standard_deviation":
            float(standard_deviation),

        # -----------------------------------------------------
        # IMAGE DIMENSIONS USED FOR THIS ANALYSIS
        # -----------------------------------------------------

        "orientation_image_width":
            int(image_width),

        "orientation_image_height":
            int(image_height),

        # -----------------------------------------------------
        # VERTICAL-LINE DIAGNOSTICS
        # -----------------------------------------------------

        "vertical_segment_count":
            int(len(vertical_segments)),

        "vertical_cluster_count":
            int(len(vertical_clusters)),

        "vertical_cluster_tolerance":
            float(cluster_tolerance),

        "vertical_line_clusters":
            vertical_clusters,

        "vertical_cluster_gaps":
            vertical_cluster_gaps,

        "vertical_cluster_pairs":
            vertical_cluster_pairs,

        "strong_vertical_cluster_indices":
            strong_vertical_cluster_indices,

        # -----------------------------------------------------
        # DEBUG IMAGE
        #
        # main.py already removes this before JSON is returned.
        # -----------------------------------------------------

        "display":
            display
    }
