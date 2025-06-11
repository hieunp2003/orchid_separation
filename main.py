import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- Import pipeline helper functions ---
from scripts.segmentation_and_detection import (
    tta_segmentation,          # detect mask + OBB with TTA
    normal_segmentation,       # detect mask + OBB without TTA
    full_segment,              # segment full branch mask
    obb_detection_xywhr        # detect OBBs (xywhr format)
)

from scripts.find_direction import process_bud_direction          # compute direction vectors
from scripts.bud_separating_function import (
    build_buds_list,           # determine bud ordering
    separation_vector,         # compute separation vector
    single_branch_separation   # cut-line for single branch
)

from scripts.draw_function import (
    draw_obb_fixed_color,
    put_debug_text,
    draw_vector,
    draw_bud_annotations,
    draw_cut_line
)

# === MODEL PATHS ===
MODEL_SEG_PATH     = "models/best_seg.pt"
MODEL_OBB_PATH     = "models/best_obb.pt"
MODEL_SEGFULL_PATH = "models/best_segfull.pt"

# === Input / Output folders ===
test_path = r"data\test_images"
save_dir  = r"runs\casetest"
os.makedirs(save_dir, exist_ok=True)

# === Verify model files exist BEFORE loading ===
for path in [MODEL_SEG_PATH, MODEL_OBB_PATH, MODEL_SEGFULL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing model file: {path}")

# === Load YOLO models ===
model_seg     = YOLO(MODEL_SEG_PATH)
model_obb     = YOLO(MODEL_OBB_PATH)
model_segfull = YOLO(MODEL_SEGFULL_PATH)

if __name__ == "__main__":

    # --- Iterate over every image in test folder ---
    for img_file in os.listdir(test_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Load and resize image
        img_path = os.path.join(test_path, img_file)
        image    = cv2.imread(img_path)
        image    = cv2.resize(image, (640, 640))

        # ----------------------- INFERENCE -----------------------

        # 1️⃣ Segment full branch mask
        big_masks = full_segment(image, model_segfull)

        # 2️⃣ Bud segmentation + TTA
        masked_image, darken_masks, bud_masks, selected_big_mask = tta_segmentation(
            image, big_masks, model_seg
        )
        print("Selected big masks:", len(selected_big_mask))
        print("Darken masks:", len(darken_masks))
        print("Bud masks:", len(bud_masks))

        # 3️⃣ OBB detection
        bud_obbs, darken_obbs = obb_detection_xywhr(
            image, selected_big_mask, model_obb, conf_thres=0.295
        )
        print("Darken OBBs:", len(darken_obbs))
        print("Bud OBBs:", len(bud_obbs))

        # 4️⃣ Compute direction vectors from each OBB to nearest mask centroid
        direction_vectors, branches_centroids, darken_centroid = process_bud_direction(
            bud_obbs, darken_obbs, darken_masks, bud_masks
        )

        # 5️⃣ Determine bud ordering
        buds_list = build_buds_list(direction_vectors, darken_centroid)

        # 6️⃣ Compute cut-lines
        if len(bud_obbs) == 1:                                  # Single bud on branch
            if not direction_vectors:
                raise ValueError("direction_vectors empty for single bud case.")
            M = direction_vectors[0][0]

            leaf_line, darken_line = single_branch_separation(bud_obbs, M)
            # Draw leaf cut (green)
            masked_image = draw_cut_line(masked_image, leaf_line, color=(0, 255, 0))
            # Draw darken cut (red)
            masked_image = draw_cut_line(masked_image, darken_line, color=(0, 0, 255))

        else:                                                    # Multiple buds
            # 6.1 Compute separation vector / line
            sep_vec = separation_vector(buds_list, darken_centroid)

            # 6.2 Draw separation line
            if isinstance(sep_vec, list):
                # If list of points, connect consecutive points
                if len(sep_vec) >= 2 and all(len(p) == 2 for p in sep_vec):
                    for i in range(len(sep_vec) - 1):
                        masked_image = draw_cut_line(
                            masked_image, [sep_vec[i], sep_vec[i + 1]], (0, 120, 255)
                        )
                    put_debug_text(masked_image, "list>=2")
                else:
                    print("⚠️ Invalid sep_vec:", sep_vec)
                    put_debug_text(masked_image, "MISSING POINTS")
            else:
                print("⚠️ sep_vec not list:", sep_vec)
                put_debug_text(masked_image, "INVALID")

        # ------------------- VISUALIZATION ---------------------

        # Draw OBBs
        masked_image = draw_obb_fixed_color(bud_obbs, masked_image)

        # Draw direction vectors
        for M, dir_vec, _, length, _ in direction_vectors:
            end = M + dir_vec * length
            draw_vector(masked_image, M, end)

        # Draw darken centroid
        x_darken, y_darken = darken_centroid
        cv2.circle(masked_image, (int(x_darken), int(y_darken)), 3, (255, 255, 255), -1)

        # Draw branch centroids
        for cx, cy in (c.astype(int) for c in branches_centroids):
            cv2.circle(masked_image, (cx, cy), 3, (0, 0, 255), -1)

        # Draw bud order annotations
        masked_image = draw_bud_annotations(masked_image, buds_list)

        # 8️⃣ Save result
        save_path = os.path.join(save_dir, img_file)
        cv2.imwrite(save_path, masked_image)
        print(f"✅ Saved: {save_path}")

        # 9️⃣ Display (ESC to exit)
        cv2.imshow("Seg + OBB + Dir", masked_image)
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()
