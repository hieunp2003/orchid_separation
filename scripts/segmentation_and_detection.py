
import cv2
import numpy as np
import math
import random
from scripts.find_direction import is_point_inside_mask


def apply_tta(image):
    return {
        'original': image.copy(),
        'vflip': cv2.flip(image, 0),
        'hflip': cv2.flip(image, 1),
    }

def de_transform_mask(mask, tta_type):
    if tta_type == 'vflip':
        return cv2.flip(mask, 0)
    elif tta_type == 'hflip':
        return cv2.flip(mask, 1)
    return mask

def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union != 0 else 0

def is_mask_inside_big_mask(mask_small, mask_big, ratio_thresh=0.5):
    if mask_small.shape != mask_big.shape:
        raise ValueError("Shapes must match")
    overlap = np.logical_and(mask_small, mask_big).sum()
    area_small = np.sum(mask_small)
    if area_small == 0:
        return False
    return overlap / area_small >= ratio_thresh

def find_big_mask_with_fewest_buds(big_masks, bud_masks, darken_masks, ratio_thresh=0.5):
    """
    Find the big_mask containing the fewest buds, dựa trên IoU giữa từng bud_mask và big_mask.

    Returns:
        selected_big_mask: np.ndarray (1, H, W)
        selected_bud_masks: list of np.ndarray
        selected_darken_masks: list of np.ndarray
    """

    # Count the number of buds in each big_mask
    count_per_big = []
    for bmask in big_masks:
        count = 0
        for bud in bud_masks:
            if is_mask_inside_big_mask(bud, bmask, ratio_thresh):
                count += 1
        count_per_big.append(count)

    # Select the big_mask with the fewest buds
    selected_index = int(np.argmin(count_per_big))
    selected_big_mask = big_masks[selected_index].copy().astype(np.uint8)

    # Filter bud & darken masks inside selected big_mask
    selected_bud_masks = [m for m in bud_masks if is_mask_inside_big_mask(m, selected_big_mask, ratio_thresh)]
    selected_darken_masks = [m for m in darken_masks if is_mask_inside_big_mask(m, selected_big_mask, ratio_thresh)]

    return selected_big_mask, selected_bud_masks, selected_darken_masks

#------------------------------------------------------------------------------------------------
# apply TTA find selected_big_mask, selected_bud_masks, selected_darken_masks
def tta_segmentation(image, big_masks, model, conf=0.595):
    vis = image.copy()
    merged_masks = []
    merged_classes = []
    bud_masks = []
    darken_masks = []

    for tta_name, aug_img in apply_tta(image).items():
        result = model.predict(aug_img, conf=conf, save=False, device='cpu')[0]
        if result.masks is None:
            continue

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for m, cls_id in zip(masks, classes):
            m = de_transform_mask(m, tta_name)

            matched = False
            for i, existing in enumerate(merged_masks):
                if mask_iou(m, existing) > 0.5 and merged_classes[i] == cls_id:
                    merged_masks[i] = np.logical_or(merged_masks[i], m)
                    matched = True
                    break
            if not matched:
                merged_masks.append(m)
                merged_classes.append(cls_id)

    for m, cls_id in zip(merged_masks, merged_classes):
        if cls_id == 0:
            darken_masks.append(m)
        if cls_id == 1:
            bud_masks.append(m)

    # Select the branch with the fewest buds
    selected_big_mask, selected_bud_masks, selected_darken_masks = find_big_mask_with_fewest_buds(
        big_masks, bud_masks, darken_masks
    )

    mask_canvas = np.zeros_like(image)
    # draw bud masks
    for mask in selected_bud_masks:
        color = [random.randint(100, 255) for _ in range(3)]
        for c in range(3):
            mask_canvas[:, :, c] = np.where(mask == 1, color[c], mask_canvas[:, :, c])
        #draw black contour for bud_mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_canvas, contours, -1, (0, 0, 0), thickness=2)
    # draw darken masks
    for mask in selected_darken_masks:
        for c in range(3):
            mask_canvas[:, :, c] = np.where(mask == 1, 0, mask_canvas[:, :, c])
        # draw red contour for darken_mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_canvas, contours, -1, (0, 0, 255), thickness=2)
    # Blend original image with mask; opacity 0.5)
    blended = cv2.addWeighted(vis, 0.5, mask_canvas, 0.5, 0)

    return blended, selected_darken_masks, selected_bud_masks, selected_big_mask
# without TTA
def normal_segmentation(image, big_masks, model, conf=0.595):
    vis = image.copy()
    bud_masks = []
    darken_masks = []

    # Run segmentation
    result = model.predict(image, conf=conf, save=False, device='cpu')[0]
    if result.masks is None:
        print("⚠️ No masks detected.")
        return vis, [], [], None

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for m, cls_id in zip(masks, classes):
        m = (m > 0.5).astype(np.uint8)
        if cls_id == 0:
            darken_masks.append(m)
        elif cls_id == 1:
            bud_masks.append(m)

    # Find the big_mask with the fewest buds
    selected_big_mask, selected_bud_masks, selected_darken_masks = find_big_mask_with_fewest_buds(
        big_masks, bud_masks, darken_masks
    )

    # === Visualization ===
    mask_canvas = np.zeros_like(image)

    for mask in selected_bud_masks:
        color = [random.randint(100, 255) for _ in range(3)]
        for c in range(3):
            mask_canvas[:, :, c] = np.where(mask == 1, color[c], mask_canvas[:, :, c])
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_canvas, contours, -1, (0, 0, 0), thickness=2)

    for mask in selected_darken_masks:
        for c in range(3):
            mask_canvas[:, :, c] = np.where(mask == 1, 0, mask_canvas[:, :, c])
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_canvas, contours, -1, (0, 0, 255), thickness=2)

    blended = cv2.addWeighted(vis, 0.5, mask_canvas, 0.5, 0)

    return blended, selected_darken_masks, selected_bud_masks, selected_big_mask
# find bud_obb, darken_obb
def obb_detection_xywhr(image, big_mask, model, conf_thres=0.5):
    """
    Detect OBBs from YOLO and filter boxes class 0 (bud) and 1 (darken) have centroid inside big_mask.

    Parameters:
        image: input image (numpy array)
        big_mask: binary mask (H, W) – the retained region
        model: YOLO model đã load (YOLOv8 OBB)
        conf_thres: confidence threshold (float)

    Returns:
        bud_detections: list of boxes with class 0 [x, y, w, h, angle, 0]
        darken_detections: list of boxes with class 1 [x, y, w, h, angle, 1]
    """
    bud_detections = []
    darken_detections = []

    result = model.predict(image, conf=conf_thres, iou=0.35, imgsz=640, verbose=False)[0]

    if result.obb and result.obb.xywhr is not None:
        xywhr = result.obb.xywhr.cpu().numpy()   # shape (N, 5): x, y, w, h, angle
        classes = result.obb.cls.cpu().numpy().astype(int)
        confs = result.obb.conf.cpu().numpy()    # shape (N,)

        for (x, y, w, h, angle), cls_id, conf in zip(xywhr, classes, confs):
            if is_point_inside_mask((x, y), big_mask):
                box = [float(x), float(y), float(w), float(h), float(angle), cls_id]
                if cls_id == 1:
                    bud_detections.append(box)
                elif cls_id == 0:
                    darken_detections.append(box)
        if len(bud_detections) == 0: print("No bud detected")
        print(len(bud_detections))
    return bud_detections, darken_detections
#---------------------------------------------------------------------------------------
#find bigmask
def full_segment(image, model_segfull, conf_thres=0.866, imgsz=640):
        result = model_segfull.predict(image, conf=conf_thres, save=False, device='cpu')[0]
        if result.masks is None:
            print("⚠️ No masks detected.")
            return None
        masks = result.masks.data.cpu().numpy()

        return masks


