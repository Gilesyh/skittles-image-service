import cv2
import numpy as np
import time


def analyse_orientation(image):
    """
    Estimate the rotation required to make the scorecard horizontal.

    Also analyse vertical scorecard ruling lines so downstream processing
    can identify the Home/Away divider without requiring a second
    OpenCV / Render request.

    Parameters
    ----------
    image : numpy.ndarray
        OpenCV image.

    Returns
    -------
    dict
        Orientation analysis plus vertical-line diagnostics.
    """

    # =========================================================
    # 1. EXISTING ORIENTATION / SKEW ANALYSIS
    # =========================================================

    # Create a copy that we'll draw our detected lines on.
    display = image.copy()

    # Convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_height, image_width = gray.shape[:2]

    # Detect edges
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # Detect line segments for orientation analysis
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

        # Keep only long-ish horizontal lines.
        #
        # IMPORTANT:
        # This is deliberately unchanged from the version that was
        # already giving good skew results.
        if abs(angle) < 20 and length > 150:

            accepted += 1
            angles.append(angle)

            # Draw accepted horizontal lines in green
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

    # =========================================================
    # 2. VERTICAL RULE ANALYSIS
    #
    # This is a separate Hough pass over the SAME edge image.
    #
    # It is deliberately more permissive than the orientation pass.
    # We want to detect fragments of vertical printed table rules,
    # because perspective, handwriting, shadows and photography can
    # break one physical rule into several detected segments.
    # =========================================================

    # Scale thresholds to image size so that this behaves sensibly
    # on differently sized photographs.
    min_vertical_length = max(
        35,
        int(image_height * 0.045)
    )

    vertical_max_gap = max(
        12,
        int(image_height * 0.025)
    )

    vertical_threshold = max(
        30,
        int(image_height * 0.045)
    )

    vertical_lines_raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=vertical_threshold,
        minLineLength=min_vertical_length,
        maxLineGap=vertical_max_gap
    )

    vertical_segments = []

    if vertical_lines_raw is not None:

        for line in vertical_lines_raw:

            if len(line) == 1:
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line

            dx = x2 - x1
            dy = y2 - y1

            length = float(
                np.hypot(dx, dy)
            )

            if length <= 0:
                continue

            # Angle relative to horizontal
            angle = float(
                np.degrees(
                    np.arctan2(dy, dx)
                )
            )

            # Normalise to -90..90
            if angle > 90:
                angle -= 180

            if angle < -90:
                angle += 180

            # Distance from perfect vertical.
            vertical_deviation = abs(
                90.0 - abs(angle)
            )

            # Allow some perspective / residual skew.
            #
            # We are intentionally not demanding a perfectly vertical
            # line here.
            if vertical_deviation > 12.0:
                continue

            # Ensure top/bottom ordering is consistent
            if y1 <= y2:
                top_x = int(x1)
                top_y = int(y1)
                bottom_x = int(x2)
                bottom_y = int(y2)
            else:
                top_x = int(x2)
                top_y = int(y2)
                bottom_x = int(x1)
                bottom_y = int(y1)

            midpoint_x = (
                top_x + bottom_x
            ) / 2.0

            vertical_segments.append({
                "top_x": top_x,
                "top_y": top_y,
                "bottom_x": bottom_x,
                "bottom_y": bottom_y,

                "midpoint_x": float(midpoint_x),

                "length": float(length),

                "angle": float(angle),

                "vertical_deviation":
                    float(vertical_deviation)
            })

    # =========================================================
    # 3. CLUSTER VERTICAL SEGMENTS BY X POSITION
    #
    # A single printed divider may have been detected as several
    # pieces. We therefore group nearby X positions into one
    # candidate physical line.
    # =========================================================

    cluster_tolerance = max(
        6.0,
        image_width * 0.008
    )

    # Longest fragments first.
    #
    # This makes strong physical rules establish clusters before
    # tiny fragments are considered.
    vertical_segments_sorted = sorted(
        vertical_segments,
        key=lambda segment: segment["length"],
        reverse=True
    )

    clusters = []

    for segment in vertical_segments_sorted:

        segment_x = segment["midpoint_x"]

        nearest_cluster = None
        nearest_distance = None

        for cluster in clusters:

            distance = abs(
                segment_x - cluster["x"]
            )

            if (
                distance <= cluster_tolerance
                and (
                    nearest_distance is None
                    or distance < nearest_distance
                )
            ):
                nearest_cluster = cluster
                nearest_distance = distance

        if nearest_cluster is None:

            clusters.append({
                "x": float(segment_x),
                "segments": [segment]
            })

        else:

            nearest_cluster["segments"].append(
                segment
            )

            # Recalculate cluster X using segment length as a weight.
            total_weight = sum(
                item["length"]
                for item in nearest_cluster["segments"]
            )

            if total_weight > 0:

                nearest_cluster["x"] = float(
                    sum(
                        item["midpoint_x"]
                        * item["length"]
                        for item
                        in nearest_cluster["segments"]
                    )
                    / total_weight
                )

    # =========================================================
    # 4. BUILD CLUSTER DIAGNOSTICS
    # =========================================================

    vertical_clusters = []

    for cluster in clusters:

        cluster_segments = cluster["segments"]

        if not cluster_segments:
            continue

        top_segment = min(
            cluster_segments,
            key=lambda segment: segment["top_y"]
        )

        bottom_segment = max(
            cluster_segments,
            key=lambda segment: segment["bottom_y"]
        )

        top_y = min(
            segment["top_y"]
            for segment in cluster_segments
        )

        bottom_y = max(
            segment["bottom_y"]
            for segment in cluster_segments
        )

        vertical_span = max(
            0,
            bottom_y - top_y
        )

        longest_segment = max(
            segment["length"]
            for segment in cluster_segments
        )

        total_detected_length = sum(
            segment["length"]
            for segment in cluster_segments
        )

        x = float(cluster["x"])

        x_fraction = (
            x / image_width
            if image_width > 0
            else 0.0
        )

        vertical_span_fraction = (
            vertical_span / image_height
            if image_height > 0
            else 0.0
        )

        longest_segment_fraction = (
            longest_segment / image_height
            if image_height > 0
            else 0.0
        )

        # A deliberately broad corridor.
        #
        # This does NOT decide that a line is the divider.
        # It simply tells downstream code which detected rules are
        # plausibly in the middle region of the photographed card.
        in_central_corridor = (
            0.30 <= x_fraction <= 0.70
        )

        # -----------------------------------------------------
        # Coverage analysis
        #
        # Merge overlapping Y intervals. This is important because
        # one printed vertical rule may appear as many fragments.
        # Simply summing their lengths would double-count overlaps.
        # -----------------------------------------------------

        intervals = sorted(
            [
                (
                    int(segment["top_y"]),
                    int(segment["bottom_y"])
                )
                for segment in cluster_segments
            ],
            key=lambda interval: interval[0]
        )

        merged_intervals = []

        for start_y, end_y in intervals:

            if not merged_intervals:

                merged_intervals.append(
                    [start_y, end_y]
                )

                continue

            previous = merged_intervals[-1]

            # Allow a modest gap between fragments of the same rule.
            merge_gap = max(
                8,
                int(image_height * 0.018)
            )

            if start_y <= previous[1] + merge_gap:

                previous[1] = max(
                    previous[1],
                    end_y
                )

            else:

                merged_intervals.append(
                    [start_y, end_y]
                )

        merged_vertical_coverage = sum(
            max(0, end_y - start_y)
            for start_y, end_y
            in merged_intervals
        )

        merged_vertical_coverage_fraction = (
            merged_vertical_coverage / image_height
            if image_height > 0
            else 0.0
        )

        # -----------------------------------------------------
        # How much of the central BODY of the image is occupied?
        #
        # The Home/Away divider on a YSSL card should repeatedly
        # appear through the table body rather than existing only
        # as one isolated line fragment.
        # -----------------------------------------------------

        body_top = int(
            image_height * 0.18
        )

        body_bottom = int(
            image_height * 0.90
        )

        body_height = max(
            1,
            body_bottom - body_top
        )

        body_coverage = 0

        for start_y, end_y in merged_intervals:

            clipped_start = max(
                start_y,
                body_top
            )

            clipped_end = min(
                end_y,
                body_bottom
            )

            if clipped_end > clipped_start:

                body_coverage += (
                    clipped_end - clipped_start
                )

        body_coverage_fraction = (
            body_coverage / body_height
        )

        # Count which broad vertical zones contain evidence.
        #
        # This gives us another way to distinguish a true structural
        # rule from a random vertical stroke of handwriting.
        zone_count = 6

        occupied_zones = 0

        for zone_index in range(zone_count):

            zone_top = int(
                body_top
                + (
                    body_height
                    * zone_index
                    / zone_count
                )
            )

            zone_bottom = int(
                body_top
                + (
                    body_height
                    * (zone_index + 1)
                    / zone_count
                )
            )

            zone_has_line = False

            for start_y, end_y in merged_intervals:

                overlap = max(
                    0,
                    min(end_y, zone_bottom)
                    - max(start_y, zone_top)
                )

                if overlap >= max(
                    5,
                    int(
                        (zone_bottom - zone_top)
                        * 0.12
                    )
                ):
                    zone_has_line = True
                    break

            if zone_has_line:
                occupied_zones += 1

        zone_coverage_fraction = (
            occupied_zones / zone_count
        )

        # -----------------------------------------------------
        # Structural score
        #
        # This is diagnostic rather than a final divider decision.
        #
        # Downstream n8n can use the ranked candidates while we
        # inspect real YSSL examples.
        # -----------------------------------------------------

        centre_distance = abs(
            x_fraction - 0.5
        )

        centre_score = max(
            0.0,
            1.0 - (
                centre_distance / 0.25
            )
        )

        span_score = min(
            1.0,
            vertical_span_fraction / 0.55
        )

        coverage_score = min(
            1.0,
            body_coverage_fraction / 0.55
        )

        zone_score = min(
            1.0,
            zone_coverage_fraction
        )

        segment_count_score = min(
            1.0,
            len(cluster_segments) / 5.0
        )

        structural_score = (
            centre_score * 0.25
            + span_score * 0.20
            + coverage_score * 0.30
            + zone_score * 0.20
            + segment_count_score * 0.05
        )

        vertical_clusters.append({
            "x": float(x),

            "x_fraction":
                float(x_fraction),

            "representative_top_x":
                int(top_segment["top_x"]),

            "representative_bottom_x":
                int(bottom_segment["bottom_x"]),

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
                int(len(cluster_segments)),

            "total_detected_length":
                float(total_detected_length),

            "merged_vertical_coverage":
                int(merged_vertical_coverage),

            "merged_vertical_coverage_fraction":
                float(
                    merged_vertical_coverage_fraction
                ),

            "body_coverage":
                int(body_coverage),

            "body_coverage_fraction":
                float(body_coverage_fraction),

            "occupied_body_zones":
                int(occupied_zones),

            "body_zone_count":
                int(zone_count),

            "zone_coverage_fraction":
                float(zone_coverage_fraction),

            "centre_distance_fraction":
                float(centre_distance),

            "structural_score":
                float(structural_score),

            "in_central_corridor":
                bool(in_central_corridor)
        })

    # =========================================================
    # 5. SORT VERTICAL CLUSTERS
    #
    # Keep all clusters in the output, but rank central structural
    # candidates first. This makes the diagnostics much easier to
    # interpret when testing real cards.
    # =========================================================

    vertical_clusters.sort(
        key=lambda cluster: (
            cluster["in_central_corridor"],
            cluster["structural_score"],
            cluster["body_coverage_fraction"],
            cluster["vertical_span_fraction"],
            cluster["segment_count"]
        ),
        reverse=True
    )

    # =========================================================
    # 6. PROVIDE A BEST CENTRAL CANDIDATE
    #
    # IMPORTANT:
    # We are NOT yet telling n8n blindly to crop here.
    #
    # This exposes what OpenCV currently believes is the strongest
    # structural candidate so we can validate it against Bell Sharks
    # and the other YSSL cards before allowing it to control crops.
    # =========================================================

    central_candidates = [
        cluster
        for cluster in vertical_clusters
        if cluster["in_central_corridor"]
    ]

    best_vertical_candidate = (
        central_candidates[0]
        if central_candidates
        else None
    )

    # =========================================================
    # 7. SAVE DEBUG IMAGE
    # =========================================================

    # Draw all vertical clusters.
    #
    # Central candidates are drawn more prominently.
    for cluster in vertical_clusters:

        x = int(
            round(cluster["x"])
        )

        top_y = int(
            cluster["top_y"]
        )

        bottom_y = int(
            cluster["bottom_y"]
        )

        if cluster["in_central_corridor"]:

            cv2.line(
                display,
                (x, top_y),
                (x, bottom_y),
                (255, 0, 255),
                3
            )

        else:

            cv2.line(
                display,
                (x, top_y),
                (x, bottom_y),
                (255, 255, 0),
                1
            )

    # Draw strongest central candidate in red.
    if best_vertical_candidate is not None:

        best_x = int(
            round(
                best_vertical_candidate["x"]
            )
        )

        cv2.line(
            display,
            (best_x, 0),
            (best_x, image_height - 1),
            (0, 0, 255),
            3
        )

    filename = (
        f"debug_lines_{int(time.time())}.jpg"
    )

    cv2.imwrite(
        filename,
        display
    )

    # =========================================================
    # 8. CONSOLE DIAGNOSTICS
    # =========================================================

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
    print("Vertical analysis")
    print("--------------------------------------")
    print(
        f"Segments     : "
        f"{len(vertical_segments)}"
    )
    print(
        f"Clusters     : "
        f"{len(vertical_clusters)}"
    )
    print(
        f"Cluster tol  : "
        f"{cluster_tolerance:.2f}px"
    )

    if best_vertical_candidate is not None:

        print(
            f"Best X       : "
            f"{best_vertical_candidate['x']:.2f}"
        )

        print(
            f"Best X frac  : "
            f"{best_vertical_candidate['x_fraction']:.4f}"
        )

        print(
            f"Best score   : "
            f"{best_vertical_candidate['structural_score']:.4f}"
        )

        print(
            f"Body cover   : "
            f"{best_vertical_candidate['body_coverage_fraction']:.4f}"
        )

        print(
            f"Zones        : "
            f"{best_vertical_candidate['occupied_body_zones']}"
            f"/"
            f"{best_vertical_candidate['body_zone_count']}"
        )

    else:

        print(
            "Best X       : none"
        )

    print("--------------------------------------")
    print()

    # =========================================================
    # 9. RETURN RESULT
    # =========================================================

    result = {
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

        "vertical_detection": {
            "threshold":
                int(vertical_threshold),

            "min_line_length":
                int(min_vertical_length),

            "max_line_gap":
                int(vertical_max_gap),

            "maximum_vertical_deviation_degrees":
                12.0
        },

        "vertical_line_clusters":
            vertical_clusters,

        "best_vertical_candidate":
            best_vertical_candidate,

        "display":
            display
    }

    return result
