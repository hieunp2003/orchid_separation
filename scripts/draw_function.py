import math
import cv2
import numpy as np

def draw_bud_annotations(image, buds_list, color=(0, 255, 255)) -> np.ndarray:
    """
    Vẽ chấm tại điểm M và đánh số thứ tự tại tâm của OBB trên ảnh.

    Args:
        image: Ảnh gốc.
        buds_list: Danh sách chứa 'M' (np.ndarray), 'OBB' (np.ndarray), ...
        color: Màu vẽ số thứ tự và điểm M (mặc định: vàng).

    Returns:
        image_out: Ảnh đã được vẽ.
    """
    image_out = image.copy()

    for idx, bud in enumerate(buds_list):
        M = bud['M']
        OBB = bud['OBB']  # [cx, cy, w, h, angle_rad]

        # Vẽ chấm tại M
        M_int = tuple(map(int, M))
        cv2.circle(image_out, M_int, radius=4, color=(0, 0, 255), thickness=-1)

        # Tính center của OBB
        cx, cy = int(OBB[0]), int(OBB[1])

        # Vẽ số thứ tự tại tâm OBB
        cv2.putText(
            image_out,
            str(idx + 1),
            (cx + 5, cy - 5),  # offset cho dễ nhìn
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2
        )

    return image_out

def draw_vector(masked_image, start, end, color=(0, 255, 255), thickness=1, tip_length=0.1):
        cv2.arrowedLine(
            masked_image,
            tuple(start.astype(int)),
            tuple(end.astype(int)),
            color,
            thickness,
            tipLength=tip_length
        )

def put_debug_text(image, text):
    """
    Ghi text debug lên góc trái ảnh.

    Parameters:
        image: ảnh cần vẽ chữ (ndarray)
        text: chuỗi nội dung cần hiển thị
    """
    cv2.putText(
        image,
        text,
        (5, 20),  # Góc trái trên, dịch xuống một chút
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,             # Cỡ chữ
        (255, 255, 255), # Màu chữ (trắng)
        1,               # Độ dày nét chữ
        cv2.LINE_AA
    )

# If detections include class_id: [x, y, w, h, angle, class_id]
def draw_obb_fixed_color(detections, image):
    image_out = image.copy()

    for det in detections:
        if len(det) == 6:
            x, y, w, h, angle, class_id = det
        else:
            x, y, w, h, angle = det
            class_id = -1  # default if not provided

        # Skip invalid boxes
        if w < 1 or h < 1:
            continue

        # Fixed color
        if class_id == 0:
            color = (0, 255, 0)  # green
        elif class_id == 1:
            color = (0, 0, 255)  # red
        else:
            color = (255, 255, 255)  # white

        rect = ((x, y), (w, h), angle * 180 / math.pi)  # convert to degrees for OpenCV
        box = cv2.boxPoints(rect).astype(int)

        # Draw box
        cv2.polylines(image_out, [box], isClosed=True, color=color, thickness=2)

        # Draw label
        # cx, cy = int(x), int(y)
        # label = f'class_{class_id}' if class_id != -1 else 'object'
        # cv2.putText(image_out, label, (cx - 20, cy - 10),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)

    return image_out

def draw_cut_line(image, line, color=(0, 255, 0), thickness=2):
    pt1 = tuple(map(int, line[0]))
    pt2 = tuple(map(int, line[1]))
    cv2.line(image, pt1, pt2, color, thickness)
    return image