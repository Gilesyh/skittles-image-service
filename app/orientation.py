import cv2
import numpy as np
import time


def analyse_orientation(image):
    """
    Estimate the rotation required to make the scorecard horizontal.

    Also retain diagnostic information about near-vertical printed
    lines detected by the SAME Hough transform.

    IMPORTANT:
    - Existing horizontal/skew behaviour is preserved.
    - Vertical-line analysis is additive only.
    - No vertical line is yet treated as the authoritative YSSL split.
    - No extra HTTP/OpenCV request is required.

    Parameters
    ----------
    image : numpy.ndarray
        OpenCV image.

    Returns
    -------
    dict
        Orientation analysis plus vertical-line diagnostics.
    """

    # ---------------------------------------------------------
    # 1. BASIC IMAGE INFORMATION
    # ---------------------------------------------------------

    image_height, image_width = image.shape[:2]

    # Create a copy that we'll draw our detected lines on.
    display = image.copy()

    # ---------------------------------------------------------
    # 2. CONVERT TO GREYSCALE
    # ---------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------------------------------
    # 3. DETECT EDGES
    # ---------------------------------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # ---------------------------------------------------------
    # 4. DETECT LINE SEGMENTS
    #
    # This is the same Hough operation previously used for
    # orientation. We now retain useful vertical lines as well.
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 5. EXISTING HORIZONTAL / SKEW ANALYSIS
    # ---------------------------------------------------------

    angles = []

    accepted = 0
    rejected = 0

    # ---------------------------------------------------------
    # 6. NEW VERTICAL-LINE COLLECTION
    #
    # These are diagnostics only at this stage.
    #
    # A near-vertical line is allowed to deviate by up to
    # 20 degrees from true vertical.
    #
    # We deliberately retain its actual top/bottom X values
    # rather than assuming it is perfectly vertical.
    # ---------------------------------------------------------

    vertical_segments = []

    for line in lines:

        # OpenCV sometimes returns [[x1,y1,x2,y2]]
        # and sometimes [x1,y1,x2,y2].
        if len(line) == 1:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line

        # Convert NumPy integer types to ordinary Python ints.
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        dx = x2 - x1
        dy = y2 - y1

        length = float(
            np.hypot(dx, dy)
        )

        angle = float(
            np.degrees(
                np.arctan2(dy, dx)
            )
        )

        # Convert into range -90..90.
        if angle > 90:
            angle -= 180

        if angle < -90:
            angle += 180

        # -----------------------------------------------------
        # EXISTING HORIZONTAL TEST
        # -----------------------------------------------------

        if abs(angle) < 20 and length > 150:

            accepted += 1
            angles.append(angle)

            # Draw accepted horizontal lines in green.
            cv2.line(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        else:

            # Preserve the meaning of the existing diagnostic:
            # anything not accepted for skew measurement is
            # counted as rejected.
            rejected += 1

        # -----------------------------------------------------
        # NEW VERTICAL TEST
        #
        # A line whose absolute angle is >= 70 degrees is
        # within 20 degrees of vertical.
        # -----------------------------------------------------

        if abs(angle) >= 70 and length > 150:

            # Arrange endpoints so "top" really is the upper
            # end of the detected segment.
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

            midpoint_y = (
                top_y + bottom_y
            ) / 2.0

            vertical_segments.append({
                "top_x": int(top_x),
                "top_y": int(top_y),

                "bottom_x": int(bottom_x),
                "bottom_y": int(bottom_y),

                "midpoint_x": float(midpoint_x),
                "midpoint_y": float(midpoint_y),

                "length": float(length),
                "angle": float(angle)
            })

    # ---------------------------------------------------------
    # 7. FINISH EXISTING ORIENTATION ANALYSIS
    # ---------------------------------------------------------

    if len(angles) == 0:
        raise Exception(
            "No suitable horizontal lines found."
        )

    angles = np.array(
        angles,
        dtype=np.float64
    )

    median_angle = float(
        np.median(angles)
    )

    mean_angle = float(
        np.mean(angles)
    )

    std = float(
        np.std(angles)
    )

    # Existing confidence estimate.
    confidence = max(
        0.0,
        min(
            1.0,
            1.0 - (std / 10.0)
        )
    )

    # ---------------------------------------------------------
    # 8. CLUSTER VERTICAL SEGMENTS BY HORIZONTAL POSITION
    #
    # Hough commonly detects both edges of one printed rule,
    # and can detect the same physical rule as several separate
    # segments.
    #
    # We therefore group nearby midpoint-X positions.
    #
    # IMPORTANT:
    # These clusters are still diagnostics. We are NOT yet
    # declaring any one cluster to be the Home/Away divider.
    # ---------------------------------------------------------

    vertical_segments_sorted = sorted(
        vertical_segments,
        key=lambda item: item["midpoint_x"]
    )

    # Scale tolerance with image width while keeping sensible
    # lower/upper limits.
    cluster_tolerance = max(
        8.0,
        min(
            24.0,
            image_width * 0.008
        )
    )

    raw_clusters = []

    for segment in vertical_segments_sorted:

        segment_x = segment["midpoint_x"]

        best_cluster = None
        best_distance = None

        for cluster in raw_clusters:

            distance = abs(
                segment_x -
                cluster["current_x"]
            )

            if (
                distance <= cluster_tolerance
                and
                (
                    best_distance is None
                    or distance < best_distance
                )
            ):
                best_cluster = cluster
                best_distance = distance

        if best_cluster is None:

            raw_clusters.append({
                "segments": [segment],
                "current_x": float(segment_x)
            })

        else:

            best_cluster["segments"].append(
                segment
            )

            # Recalculate cluster X using line-length weighting.
            total_weight = sum(
                item["length"]
                for item in best_cluster["segments"]
            )

            if total_weight > 0:

                best_cluster["current_x"] = (
                    sum(
                        item["midpoint_x"]
                        * item["length"]
                        for item
                        in best_cluster["segments"]
                    )
                    / total_weight
                )

    # ---------------------------------------------------------
    # 9. SUMMARISE EACH VERTICAL CLUSTER
    # ---------------------------------------------------------

    vertical_clusters = []

    for cluster in raw_clusters:

        segments = cluster["segments"]

        if not segments:
            continue

        total_length = float(
            sum(
                segment["length"]
                for segment in segments
            )
        )

        longest_length = float(
            max(
                segment["length"]
                for segment in segments
            )
        )

        top_y = int(
            min(
                segment["top_y"]
                for segment in segments
            )
        )

        bottom_y = int(
            max(
                segment["bottom_y"]
                for segment in segments
            )
        )

        vertical_span = int(
            bottom_y - top_y
        )

        # Length-weighted representative X.
        if total_length > 0:

            representative_x = float(
                sum(
                    segment["midpoint_x"]
                    * segment["length"]
                    for segment in segments
                )
                / total_length
            )

        else:

            representative_x = float(
                np.mean([
                    segment["midpoint_x"]
                    for segment in segments
                ])
            )

        # Representative top/bottom X values are retained so
        # we can later account for perspective if required.
        top_segment = min(
            segments,
            key=lambda item: item["top_y"]
        )

        bottom_segment = max(
            segments,
            key=lambda item: item["bottom_y"]
        )

        representative_top_x = int(
            top_segment["top_x"]
        )

        representative_bottom_x = int(
            bottom_segment["bottom_x"]
        )

        x_fraction = float(
            representative_x / image_width
        )

        span_fraction = float(
            vertical_span / image_height
        )

        longest_fraction = float(
            longest_length / image_height
        )

        # Merely diagnostic:
        # is this cluster somewhere in the broad central
        # portion of the photograph?
        in_central_corridor = bool(
            0.25 <= x_fraction <= 0.75
        )

        vertical_clusters.append({
            "x": float(representative_x),

            "x_fraction": float(x_fraction),

            "representative_top_x":
                int(representative_top_x),

            "representative_bottom_x":
                int(representative_bottom_x),

            "top_y": int(top_y),
            "bottom_y": int(bottom_y),

            "vertical_span":
                int(vertical_span),

            "vertical_span_fraction":
                float(span_fraction),

            "longest_segment":
                float(longest_length),

            "longest_segment_fraction":
                float(longest_fraction),

            "segment_count":
                int(len(segments)),

            "total_detected_length":
                float(total_length),

            "in_central_corridor":
                in_central_corridor
        })

    # ---------------------------------------------------------
    # 10. SORT VERTICAL CLUSTERS
    #
    # Most useful-looking structural lines first:
    #
    # 1. greatest vertical coverage
    # 2. greatest accumulated Hough support
    # 3. greatest number of supporting segments
    #
    # This ordering is diagnostic only.
    # ---------------------------------------------------------

    vertical_clusters.sort(
        key=lambda item: (
            item["vertical_span"],
            item["total_detected_length"],
            item["segment_count"]
        ),
        reverse=True
    )

    # Avoid returning an unnecessarily huge JSON object.
    # Twenty strongest clusters is ample for diagnosis.
    vertical_clusters_returned = (
        vertical_clusters[:20]
    )

    # ---------------------------------------------------------
    # 11. DEBUG IMAGE
    #
    # Preserve the existing debug image behaviour.
    #
    # Horizontal accepted lines are green.
    # Vertical diagnostic lines are blue.
    # ---------------------------------------------------------

    for segment in vertical_segments:

        cv2.line(
            display,
            (
                int(segment["top_x"]),
                int(segment["top_y"])
            ),
            (
                int(segment["bottom_x"]),
                int(segment["bottom_y"])
            ),
            (255, 0, 0),
            2
        )

    filename = (
        f"debug_lines_{int(time.time())}.jpg"
    )

    cv2.imwrite(
        filename,
        display
    )

    # ---------------------------------------------------------
    # 12. SERVER LOG DIAGNOSTICS
    # ---------------------------------------------------------

    print()
    print("--------------------------------------")
    print("Orientation analysis")
    print("--------------------------------------")
    print(f"Debug image : {filename}")
    print(f"Image       : {image_width} x {image_height}")
    print(f"Accepted    : {accepted}")
    print(f"Rejected    : {rejected}")
    print(f"Median      : {median_angle:.2f}°")
    print(f"Mean        : {mean_angle:.2f}°")
    print(f"Std Dev     : {std:.2f}")
    print(f"Confidence  : {confidence:.2f}")
    print(
        f"Vertical segments : "
        f"{len(vertical_segments)}"
    )
    print(
        f"Vertical clusters : "
        f"{len(vertical_clusters)}"
    )

    for index, cluster in enumerate(
        vertical_clusters_returned[:10],
        start=1
    ):
        print(
            f"  V{index}: "
            f"x={cluster['x']:.1f}, "
            f"x%={cluster['x_fraction']:.3f}, "
            f"span={cluster['vertical_span']}, "
            f"segments={cluster['segment_count']}"
        )

    print("--------------------------------------")
    print()

    # ---------------------------------------------------------
    # 13. RETURN
    #
    # Existing fields remain unchanged.
    #
    # New fields are additive only.
    # Nothing downstream currently needs to consume them.
    # ---------------------------------------------------------

    return {
        # Existing output
        "rotation_angle":
            median_angle,

        "confidence":
            confidence,

        "accepted_lines":
            accepted,

        "rejected_lines":
            rejected,

        "median_angle":
            median_angle,

        "mean_angle":
            mean_angle,

        "standard_deviation":
            std,

        # New diagnostic output
        "orientation_image_width":
            int(image_width),

        "orientation_image_height":
            int(image_height),

        "vertical_segment_count":
            int(len(vertical_segments)),

        "vertical_cluster_count":
            int(len(vertical_clusters)),

        "vertical_cluster_tolerance":
            float(cluster_tolerance),

        "vertical_line_clusters":
            vertical_clusters_returned,

        # Existing non-JSON debug object.
        # main.py already removes this before returning the
        # HTTP response.
        "display":
            display
    }
