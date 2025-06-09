import numpy as np
import cv2

def is_point_inside_mask(point, mask):
    """
    Kiểm tra xem điểm (x, y) có nằm trong vùng foreground (giá trị > 0) của mask hay không.

    Parameters:
        point: (x, y)
        mask: numpy 2D array (height, width), nhị phân hoặc float

    Returns:
        True nếu điểm nằm trong mask, False nếu không.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, but got shape {mask.shape}")

    x, y = int(round(point[0])), int(round(point[1]))
    h, w = mask.shape

    if 0 <= x < w and 0 <= y < h:
        return mask[y, x] > 0  # lưu ý: hàng là y, cột là x
    return False

def get_mask_centroid(mask):
    if mask is None:
        return None
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return np.array([cx, cy])

def get_triangle_centroid(M_list) :
    """
    Tính tâm tam giác tạo bởi 3 điểm đầu tiên trong danh sách M.

    Args:
        M_list: List các điểm np.ndarray dạng (x, y)

    Returns:
        Centroid (np.ndarray) của tam giác

    Raises:
        ValueError nếu số điểm < 3
    """
    if len(M_list) < 3:
        return [320,320]

    A = np.array(M_list[0], dtype=float)
    B = np.array(M_list[1], dtype=float)
    C = np.array(M_list[2], dtype=float)

    centroid = (A + B + C) / 3.0
    return centroid

def get_bud_direction(O, OG_unit, w, h, angle_rad):
    v = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    u = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

    candidates = [
        (v, w),
        (-v, w),
        (u, h),
        (-u, h)
    ]

    best_vector, size = max(
        candidates,
        key=lambda item: abs(np.dot(OG_unit, item[0]))
    )

    # ──> make sure best_vector actually has a positive projection onto OG_unit
    if np.dot(best_vector, OG_unit) < 0:
        best_vector = -best_vector

    # now compute M
    M = O - 0.5 * size * best_vector
    N = O + 0.5 * size * best_vector

    return M, N, best_vector

def get_max_mask_length(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0
    return max(xs.max() - xs.min(), ys.max() - ys.min())

def process_bud_direction(bud_obbs, darken_obbs, darken_masks, bud_masks):
    vectors = []
    centroids = []
    M_list = []

    for mask in bud_masks:
        centroid = get_mask_centroid(mask)
        if centroid is not None:
            centroids.append(centroid)
    darken_mask = darken_masks[0] if len(darken_masks) > 0 else None
    darken_centroid = get_mask_centroid(darken_mask)

    for obb in bud_obbs:
        x, y, w, h, angle_rad, cls_id = obb
        O = np.array([x, y])

        found_mask = None

        # Mỗi mask được chứa nhiều OBBs
        for mask in bud_masks:
            if is_point_inside_mask(O, mask):
                found_mask = mask
                break

        if found_mask is None:
            continue
        # Tính bud direction theo Mask centroid
        G = get_mask_centroid(found_mask)
        if G is None:
            continue

        OG = G - O
        OG_unit = OG / (np.linalg.norm(OG) + 1e-6)

        M, N, best_vector = get_bud_direction(O, OG_unit, w, h, angle_rad)

        # Nếu M nằm ngoài mask (vector chỉ phương khả năng lỗi)
        # thì Tính bud direction theo darken_mask centroid
        if (not is_point_inside_mask(M, found_mask) or not is_point_inside_mask(N,found_mask)) and darken_mask is not None:
            C = darken_centroid
            if C is None:
                continue
            CO = O - C
            CO_unit = CO / (np.linalg.norm(CO) + 1e-6)
            M, N, best_vector = get_bud_direction(O, CO_unit, w, h, angle_rad)

        length = 0.9 * get_max_mask_length(found_mask)
        if length == 0:
            continue

        vectors.append((M, best_vector, obb, length, found_mask))
        M_list.append(M)
    if darken_centroid is None:
        if len(darken_obbs) > 0:
            x, y = darken_obbs[0][:2]
            darken_centroid = np.array([x, y])
        elif len(M_list) >= 3:
            darken_centroid = get_triangle_centroid(M_list)
        else:
            darken_centroid = np.array([25, 5])
            print("no darken_centroid")

    return vectors, centroids, darken_centroid

