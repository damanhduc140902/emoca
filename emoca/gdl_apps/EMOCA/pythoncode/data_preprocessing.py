import os
import numpy as np
from gdl_apps.EMOCA.demos.test_emoca_on_images import process_images

#{'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
expression_labels = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}


def load_data(input_folder):
    """
    Xử lý ảnh bằng EMoCA và tải dữ liệu.

    Args:
        input_folder (str): Đường dẫn đến thư mục chứa ảnh.

    Returns:
        np.array: Mảng dữ liệu đã được kết hợp từ detail, exp, và shape codes.
    """
    # Xử lý ảnh bằng EMOCA và nhận dữ liệu
    all_details, all_exps, all_shapes = process_images(input_folder)

    # Kết hợp dữ liệu
    combined_data = np.concatenate([all_details, all_exps, all_shapes], axis=1)

    return combined_data

