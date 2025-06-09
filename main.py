import os
import cv2
import numpy as np
from ultralytics import YOLO

from scripts.segmentation_and_detection import tta_segmentation # Hàm detect mask, obb
from scripts.segmentation_and_detection import normal_segmentation # no TTA
from scripts.segmentation_and_detection import full_segment
from scripts.segmentation_and_detection import obb_detection_xywhr

from scripts.find_direction import process_bud_direction # Hàm tìm vector hướng

from scripts.bud_separating_function import build_buds_list # Hàm tìm số thứ tự
from scripts.bud_separating_function import separation_vector
from scripts.bud_separating_function import single_branch_separation

from scripts.draw_function import draw_obb_fixed_color
from scripts.draw_function import put_debug_text
from scripts.draw_function import draw_vector
from scripts.draw_function import draw_bud_annotations
from scripts.draw_function import draw_cut_line

# === MODEL PATH ===
MODEL_SEG_PATH = "models/best_seg.pt"
MODEL_OBB_PATH = "models/best_obb.pt"
MODEL_SEGFULL_PATH = "models/best_segfull.pt"

# === Test folder and save folder ===
test_path = r'data\test_images'
save_dir = 'runs\casetest'
os.makedirs(save_dir, exist_ok=True)

# === Check file tồn tại trước khi load ===
for path in [MODEL_SEG_PATH, MODEL_OBB_PATH, MODEL_SEGFULL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing model file: {path}")

# === Load models ===
model_seg = YOLO(MODEL_SEG_PATH)
model_obb = YOLO(MODEL_OBB_PATH)
model_segfull = YOLO(MODEL_SEGFULL_PATH)



if __name__ == "__main__":
    # === Loop through each image ===
    for img_file in os.listdir(test_path):
        if not img_file.lower(). endswith(('.jpg', '.jpeg', '.png')):
            continue

        #load ảnh
        img_path = os.path.join(test_path, img_file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (640, 640))

        #---------------INFERENCE------------------------------

        # Step 1: Find big_mask
        big_masks = full_segment(image, model_segfull)

        # Step 2: Segmentation + TTA
        masked_image, darken_masks, bud_masks, selected_big_mask = tta_segmentation(image, big_masks, model_seg)
        print("bigmask available:", len(selected_big_mask))
        print("Number of darken masks:", len(darken_masks))
        print("Number of bud masks:", len(bud_masks))

        # Step 3: Detect OBB
        bud_obbs, darken_obbs = obb_detection_xywhr(image, selected_big_mask, model_obb, conf_thres=0.295)
        print("bigmask available:", len(selected_big_mask))
        print("Number of darken obb:", len(darken_obbs))
        print("Number of bud obb:", len(bud_obbs))
        #--------------DIRECTION---------------------------------------------------------------
        # Step 4: Find direction vectors from each OBB to nearest mask center
        direction_vectors, branches_centroids, darken_centroid = process_bud_direction(bud_obbs, darken_obbs, darken_masks, bud_masks)
        # -------------SEPARATION-------------------------------------------------------------------
        # Step 5: Find bud order
        buds_list = build_buds_list(direction_vectors, darken_centroid)

        # Step 6: Find cut line
        # Single branch
        if len(bud_obbs) == 1:
            if not direction_vectors:
                raise ValueError("bud_obbs ==1, direction_vectors is empty. Check input data or function logic.")
            M = direction_vectors[0][0]

            leaf_line, darken_line = single_branch_separation(bud_obbs, M)
            # Vẽ đường cắt lá (màu xanh dương)
            masked_image = draw_cut_line(masked_image, leaf_line, color=(0, 255, 0))
            # Vẽ đường cắt darken (màu đỏ)
            masked_image = draw_cut_line(masked_image, darken_line, color=(0, 0, 255))

        # Multiple braches
        else:
            # Step 6.1: Find separation line
            sep_vec = separation_vector(buds_list,darken_centroid)
            # Step 6.2: draw separation line
            # -> 6.2.1 nếu sep_vec trả về list thì vẽ các điểm
            if isinstance(sep_vec, list):
                # TH vẽ đường biên/ đường cắt
                if len(sep_vec) >= 2 and all(len(p) == 2 for p in sep_vec):
                    # Nối các điểm (linepoints)
                    for i in range(len(sep_vec) - 1):
                        masked_image = draw_cut_line(masked_image, [sep_vec[i], sep_vec[i + 1]], (0, 120, 255))
                    put_debug_text(masked_image, "list>=2")

                # ko đủ 2 điểm trở lên
                else:
                    print("⚠️ sep_vec không hợp lệ::", sep_vec)
                    put_debug_text(masked_image, "THIẾU ĐIỂM")

            # -> 6.2.2 nếu sep_vec không phải string hoặc list
            else:
                print("⚠️ sep_vec không hợp lệ:", sep_vec)
                put_debug_text(masked_image, "không hợp lệ")

        # -------------VISUALIZATION-------------------------------------------------------------------
        # Step 7: Draw
        # 7.0 OBB
        obb_mask_image = draw_obb_fixed_color(bud_obbs, masked_image)
        masked_image = obb_mask_image

        # 7.1 direction vectors for buds
        for M, dir_vec, _, length, _ in direction_vectors:
            end = M + dir_vec * length
            draw_vector(masked_image, M, end)

        # 7.2 darken centroid
        x_darken, y_darken = darken_centroid
        cv2.circle(masked_image, (int(x_darken),int(y_darken)), 3, (255, 255, 255), -1)

        # 7.3 braches centroid
        for i, centroid in enumerate(branches_centroids, start=1):  # Bắt đầu từ 1
            cx, cy = centroid.astype(int)
            cv2.circle(masked_image, (cx, cy), 3, (0, 0, 255), -1)

        # 7.4 bud order number
        masked_image = draw_bud_annotations(masked_image, buds_list)

        # Step 8: Save result
        save_path = os.path.join(save_dir, img_file)
        cv2.imwrite(save_path, masked_image)
        print(f"✅ Saved: {save_path}")

        # Step 9: Show image
        cv2.imshow("Seg + OBB + Dir", masked_image)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()

