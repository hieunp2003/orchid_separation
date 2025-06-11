
import cv2
import numpy as np


# ---------- Single_branch ----------
def single_branch_separation(obb, tail_point):
    """
    Sinh 2 đường cắt từ obbs và điểm đuôi M, hướng vector là MO (đi từ M lên O).

    Parameters:
        obbs: list obb hoặc 1 obb tuple (x, y, w, h, angle_rad, cls_id)
        tail_point: np.array([x, y]) – điểm đáy bud (gốc vector)

    Returns:
        cut_leaf_line: (pt1, pt2) – đường cắt giữa bud (vuông góc vector MO)
        cut_darken_line: (pt1, pt2) – đường cắt đuôi cách đáy 1/18 chiều
    """

    x, y, w, h, angle_rad, cls_id = obb[0]
    O = np.array([x, y])
    M = np.array(tail_point)
    dir_vector = O - M
    dir_unit = dir_vector / (np.linalg.norm(dir_vector) + 1e-6)

    # Vector vuông góc để vẽ đường cắt
    perp_vector = np.array([-dir_unit[1], dir_unit[0]])

    # === 1. Đường cắt lá
    bud_length = 2 / 3 * max(w, h)
    pt1_leaf = M + bud_length*dir_unit - (perp_vector * bud_length / 2)
    pt2_leaf = M + bud_length*dir_unit + (perp_vector * bud_length / 2)

    # === 2. Đường cắt đuôi (1/18 chiều từ M lùi lên trước sau)
    back_offset = 1 / 18 * max(w, h)
    darken_center = M + dir_unit * back_offset

    darken_length = 2 / 3 * max(w, h)
    pt1_darken = darken_center - (perp_vector * darken_length / 2)
    pt2_darken = darken_center + (perp_vector * darken_length / 2)

    return (pt1_leaf, pt2_leaf), (pt1_darken, pt2_darken)

# ---------- Multiple_branches ----------
def compute_angle_between_dirs(v1, v2):
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# ---------- Step 1: build bud list ----------
def build_buds_list(vectors, darken_center):
    # Tính tất cả các vector từ darken_center đến mỗi M và tính góc
    angle_list = []
    for i, (M, Dir, OBB, _, bmask) in enumerate(vectors):
        x,y,_,_,_,_ = OBB
        vec = np.array([x, y]) - darken_center
        angle = np.arctan2(vec[1], vec[0])
        angle = angle if angle >= 0 else angle + 2 * np.pi
        angle_list.append((angle, i))

    # Sắp xếp theo ngược chiều kim đồng hồ (góc tăng dần)
    angle_list.sort()
    angles_sorted = [angle for angle, _ in angle_list]
    indices_sorted = [idx for _, idx in angle_list]

    # Tìm khoảng cách góc lớn nhất giữa các điểm liên tiếp
    max_gap = -1
    max_idx = 0
    for i in range(len(angles_sorted)):
        curr_angle = angles_sorted[i]
        next_angle = angles_sorted[(i + 1) % len(angles_sorted)]
        diff = (next_angle - curr_angle) % (2 * np.pi)
        if diff > max_gap:
            max_gap = diff
            max_idx = (i + 1) % len(angles_sorted)

    # Điểm bắt đầu là nơi sau khoảng cách lớn nhất
    start_order = indices_sorted[max_idx:] + indices_sorted[:max_idx]

    buds_list = []
    for i in range(len(start_order)):
        idx = start_order[i]
        M, Dir, OBB, _, bmask = vectors[idx]
        bud = {
            'M': M,
            'Dir': Dir,
            'OBB': OBB,
            'bmask': bmask,

        }
        if i < len(start_order) - 1:
            next_idx = start_order[i + 1]
            bud['distance'] = np.linalg.norm(M - vectors[next_idx][0])
        else:
            bud['distance'] = None
        buds_list.append(bud)

    return buds_list

# ---------- Step 2: analyze separation ----------
def separation_vector(buds_list, darken_centroid):
    max_dist = -1
    selected_pair = None
    selected_dirs = None
    selected_mask = None
    w_1 =0
    w_2 =0
    selected_idx = -1
    max_dist = -1

    # Bước 1: Tìm chỉ số i tốt nhất
    for i in range(len(buds_list) - 1):
        d = buds_list[i].get('distance')
        if d is not None and d > max_dist:
            max_dist = d
            selected_idx = i

    # Bước 2: Sau vòng lặp, truy xuất dữ liệu
    if selected_idx >= 0:
        bud1 = buds_list[selected_idx]
        bud2 = buds_list[selected_idx + 1]

        M1, M2 = bud1['M'], bud2['M']
        Dir1, Dir2 = bud1['Dir'], bud2['Dir']
        bmask1, bmask2 = bud1['bmask'], bud2['bmask']
        x1, y1, w1, h1, *_ = bud1['OBB']
        x2, y2, w2, h2, *_ = bud2['OBB']

        w_1 = min(w1, h1)
        w_2 = min(w2, h2)

        # selected_pair = (M1, M2) # Option 1: cutting point considered by M1,M2
        selected_pair = ((x1,y1), (x2,y2)) # Option 2: cutting point considered by 2 obb centroid
        selected_dirs = (Dir1, Dir2)
        selected_mask = (bmask1, bmask2)

    if not selected_pair:
        return "Not enough data"

    angle = compute_angle_between_dirs(*selected_dirs)
    if angle > 120:
        # return case_11_separation(selected_pair, darken_centroid)
        return case_12_separation(selected_pair, darken_centroid, selected_mask,w_1,w_2)
    elif angle <= 120:
        # separation line is 
        return case_2_separation(*selected_pair, *selected_dirs, selected_mask,w_1,w_2)
    # else:
    #     return f"Case 3: Angle = 20°< ({angle:.2f}) ≤ 120°"


# CASEs

# Case 1
def get_overlay_centroid(mask1, mask2):
    # Ensure binary
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Intersection mask
    overlay = cv2.bitwise_and(mask1, mask2)

    # Compute moments
    M = cv2.moments(overlay)

    if M["m00"] == 0:
        return None  # No overlap

    # Centroid coordinates (cx, cy)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)
#separation line perpendicular to M1M2
def case_1_separation(selected_pair, darken_centroid):
    M1, M2 = selected_pair
    M1 = np.array(M1, dtype=float)
    M2 = np.array(M2, dtype=float)
    C = np.array(darken_centroid, dtype=float)
    length = 100
    v = M2 - M1
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("M1 and M2 must be different points.")

    # Perpendicular Vector
    v_perp = np.array([-v[1], v[0]])
    v_perp = (v_perp / np.linalg.norm(v_perp)) * length

    pt1 = C - v_perp
    pt2 = C + v_perp

    return [pt1, pt2]
#separation line go through M1M2 center
def case_11_separation(selected_pair, darken_centroid, length=100):
    M1, M2 = selected_pair
    M1 = np.array(M1, dtype=float)
    M2 = np.array(M2, dtype=float)
    C = np.array(darken_centroid, dtype=float)

    # midpoint of M1 và M2
    midpoint = (M1 + M2) / 2.0

    # Vector C -> midpoint
    direction = midpoint - C
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("darken_centroid và midpoint trùng nhau, không xác định được hướng.")

    # normalize
    unit_vector = direction / norm
    v = unit_vector * (length / 2)

    # find 2 end of cutline
    pt1 = C - v
    pt2 = C + v

    return [pt1, pt2]
#separation line go through intersection centroid
def case_12_separation(selected_pair, darken_centroid, selected_mask, w_1,w_2, length=150):
    M1, M2 = selected_pair
    M1 = np.array(M1, dtype=float)
    M2 = np.array(M2, dtype=float)
    C = np.array(darken_centroid, dtype=float)
    bmask1, bmask2 = selected_mask
    O = (w_2 * M1 + w_1 * M2) / (w_1 + w_2)

    # Trung điểm giữa mask1 và mask2
    midpoint = get_overlay_centroid(bmask1, bmask2)
    if midpoint is None:
        print("Case 1: intersection is None")
        # Nếu không giao nhau, dùng trung điểm M1 và M2
        midpoint = O
    else: print("Case 1: midpoint available ")
    midpoint = np.array(midpoint, dtype=float)

    # Vector hướng từ C đến midpoint
    direction = midpoint - C
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("darken_centroid và midpoint trùng nhau, không xác định được hướng.")

    # Vector đơn vị nhân với độ dài
    unit_vector = direction / norm
    v = unit_vector * (length / 2)

    # Tạo đoạn thẳng đi qua C, kéo dài về hai phía
    pt1 = C - v
    pt2 = C + v

    return [pt1, pt2]
#--------------------------------------------------------------

# Case 2
#find unit vector
def unit(v):

    return v / np.linalg.norm(v)
#find bisector_vector
def bisector_vector(v1, v2):
    return unit(unit(v1) + unit(v2))
#separation line parallel with bisector and go through O
def case_2_separation(M1, M2, v1, v2, selected_mask, w_1, w_2, L=150):
    bmask1,bmask2 = selected_mask
    M1 = np.array(M1, dtype=float)
    M2 = np.array(M2, dtype=float)
    O = (w_2 * M1 + w_1 * M2) / (w_1 + w_2) #point between M1, M2

    # Step 2: angle bisector vector at E
    b_vec = bisector_vector(v1, v2)

    # Step 3: intersection point H
# Option 1: take H = O
    H = O
    print("Case 2")

# Option 2: centroid of overlay_mask
    # #take H = overlay_centroid
    # H = get_overlay_centroid(bmask1, bmask2)
    # if H is None or (np.linalg.norm((M1 + M2)/2 - np.array(H, dtype=float)) > 50):
    #     print("Case 2: H is None")
    #     H = O
    #     # H = line_intersection(E, b_vec, M1, M2 - M1)
    # else: print("Case 2: H is available")

    # Step 4: separation line from H along bisector
    half_L = L / 2
    P1 = H - half_L * b_vec
    P2 = H + half_L * b_vec

    return [P1, P2]
