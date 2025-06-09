import math
import cv2
import numpy as np


def draw_bud_annotations(image, buds_list, color=(0, 255, 255)) -> np.ndarray:
    """
    Draw a dot at every M point and write the bud index at the OBB center.

    Args:
        image: Original image (numpy array).
        buds_list: List of buds; each item contains keys
                   'M' (np.ndarray) and 'OBB' (np.ndarray).
        color: Text/dot color (default: yellow).

    Returns:
        image_out: Image with annotations drawn.
    """
    image_out = image.copy()

    for idx, bud in enumerate(buds_list):
        M   = bud["M"]
        OBB = bud["OBB"]  # [cx, cy, w, h, angle_rad]

        # Draw dot at M
        M_int = tuple(map(int, M))
        cv2.circle(image_out, M_int, radius=4, color=(0, 0, 255), thickness=-1)

        # Compute OBB center
        cx, cy = int(OBB[0]), int(OBB[1])

        # Draw bud index at OBB center
        cv2.putText(
            image_out,
            str(idx + 1),
            (cx + 5, cy - 5),  # small offset for readability
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )

    return image_out


def draw_vector(masked_image,start,end,color=(0, 255, 255),thickness=1,tip_length=0.1,):
    """Draw an arrow from start to end."""
    cv2.arrowedLine(
        masked_image,
        tuple(start.astype(int)),
        tuple(end.astype(int)),
        color,
        thickness,
        tipLength=tip_length,
    )


def put_debug_text(image, text):
    """
    Render debug text at the top-left corner of an image.

    Parameters:
        image: Target image (ndarray).
        text:  Message string to display.
    """
    cv2.putText(
        image,
        text,
        (5, 20),            # top-left corner with slight offset
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,                # font size
        (255, 255, 255),    # white text
        1,                  # thickness
        cv2.LINE_AA,
    )


def draw_obb_fixed_color(detections, image):
    """
    Draw oriented bounding boxes with fixed color coding.

    If detections include class_id:
        [x, y, w, h, angle, class_id]
    Else:
        [x, y, w, h, angle]
    """
    image_out = image.copy()

    for det in detections:
        if len(det) == 6:
            x, y, w, h, angle, class_id = det
        else:
            x, y, w, h, angle = det
            class_id = -1  # default when class is not provided

        # Skip invalid boxes
        if w < 1 or h < 1:
            continue

        # Choose a fixed color by class
        if class_id == 0:
            color = (0, 255, 0)      # green
        elif class_id == 1:
            color = (0, 0, 255)      # red
        else:
            color = (255, 255, 255)  # white

        # Convert angle to degrees for OpenCV
        rect = ((x, y), (w, h), angle * 180 / math.pi)
        box  = cv2.boxPoints(rect).astype(int)

        cv2.polylines(image_out, [box], isClosed=True, color=color, thickness=2)

    return image_out


def draw_cut_line(image, line, color=(0, 255, 0), thickness=2):
    """Draw a straight line given two endpoints."""
    pt1 = tuple(map(int, line[0]))
    pt2 = tuple(map(int, line[1]))
    cv2.line(image, pt1, pt2, color, thickness)
    return image
