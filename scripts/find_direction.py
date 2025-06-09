import numpy as np
import cv2


def is_point_inside_mask(point, mask):
    """
    Check whether a point (x, y) lies inside the foreground area (value > 0)
    of a binary mask.

    Args:
        point: (x, y) tuple or array.
        mask:  2-D NumPy array (H, W), binary or float.

    Returns:
        True  – point is inside mask
        False – point is outside mask
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, received shape {mask.shape}")

    x, y = int(round(point[0])), int(round(point[1]))
    h, w = mask.shape
    if 0 <= x < w and 0 <= y < h:
        # note: row is y, column is x
        return mask[y, x] > 0
    return False


def get_mask_centroid(mask):
    """Return centroid (cx, cy) of a binary mask, or None if empty."""
    if mask is None:
        return None
    m = cv2.moments(mask.astype(np.uint8))
    if m["m00"] == 0:
        return None
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def get_triangle_centroid(M_list):
    """
    Centroid of the triangle formed by the first three points in M_list.

    Args:
        M_list: list of np.ndarray points (x, y)

    Returns:
        np.ndarray centroid.  If fewer than 3 points, returns [320, 320].
    """
    if len(M_list) < 3:
        return [320, 320]
    A, B, C = (np.array(p, dtype=float) for p in M_list[:3])
    return (A + B + C) / 3.0


def get_bud_direction(O, OG_unit, w, h, angle_rad):
    """
    Choose the best edge direction (v or u) aligned with OG_unit.

    Returns:
        M, N  – two end-points of the chosen edge
        best_vector – unit direction vector
    """
    v = np.array([np.cos(angle_rad),  np.sin(angle_rad)])
    u = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

    candidates = [(v, w), (-v, w), (u, h), (-u, h)]
    best_vector, size = max(
        candidates,
        key=lambda item: abs(np.dot(OG_unit, item[0]))
    )

    # ensure positive projection onto OG_unit
    if np.dot(best_vector, OG_unit) < 0:
        best_vector = -best_vector

    M = O - 0.5 * size * best_vector
    N = O + 0.5 * size * best_vector
    return M, N, best_vector


def get_max_mask_length(mask):
    """Return max side length of the bounding box of a mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0
    return max(xs.max() - xs.min(), ys.max() - ys.min())


def process_bud_direction(bud_obbs, darken_obbs, darken_masks, bud_masks):
    """
    For each bud OBB, compute a direction vector and related info.

    Returns:
        vectors     – list of tuples (M, dir_vec, obb, length, mask)
        centroids   – list of bud mask centroids
        darken_centroid – centroid of darken mask or fallback location
    """
    vectors, centroids, M_list = [], [], []

    # Gather centroids of every bud mask
    for mask in bud_masks:
        c = get_mask_centroid(mask)
        if c is not None:
            centroids.append(c)

    darken_mask      = darken_masks[0] if darken_masks else None
    darken_centroid  = get_mask_centroid(darken_mask)

    for obb in bud_obbs:
        x, y, w, h, angle_rad, cls_id = obb
        O = np.array([x, y])

        # find bud mask containing this OBB center
        found_mask = next((m for m in bud_masks if is_point_inside_mask(O, m)), None)
        if found_mask is None:
            continue

        # primary direction based on mask centroid
        G = get_mask_centroid(found_mask)
        if G is None:
            continue

        OG_unit = (G - O) / (np.linalg.norm(G - O) + 1e-6)
        M, N, best_vec = get_bud_direction(O, OG_unit, w, h, angle_rad)

        # fallback: use darken centroid if M or N lies outside mask
        if (not is_point_inside_mask(M, found_mask) or
            not is_point_inside_mask(N, found_mask)) and darken_mask is not None:

            C = darken_centroid
            if C is None:
                continue
            CO_unit = (O - C) / (np.linalg.norm(O - C) + 1e-6)
            M, N, best_vec = get_bud_direction(O, CO_unit, w, h, angle_rad)

        length = 0.9 * get_max_mask_length(found_mask)
        if length == 0:
            continue

        vectors.append((M, best_vec, obb, length, found_mask))
        M_list.append(M)

    # Estimate darken centroid if missing
    if darken_centroid is None:
        if darken_obbs:
            x, y = darken_obbs[0][:2]
            darken_centroid = np.array([x, y])
        elif len(M_list) >= 3:
            darken_centroid = get_triangle_centroid(M_list)
        else:
            darken_centroid = np.array([25, 5])
            print("No darken_centroid determined")

    return vectors, centroids, darken_centroid
