import joblib
import numpy as np
from tensorflow.keras.models import load_model

def load_saved_models(model_dir, model_version="91"):
    """
    Tải các mô hình đã được lưu trữ.

    Args:
        model_dir (str): Đường dẫn đến thư mục chứa các mô hình đã lưu.
        model_version (str): Phiên bản của các mô hình để tải.

    Returns:
        dict: Một từ điển chứa các mô hình đã tải.
    """
    models = {}
    models['random_forest'] = joblib.load(f"{model_dir}/random_forest_model_{model_version}.joblib")
    models['svm'] = joblib.load(f"{model_dir}/svm_model_{model_version}.joblib")
    models['gboost'] = joblib.load(f"{model_dir}/gboost_model_{model_version}.joblib")
    models['nn'] = load_model(f"{model_dir}/nn_model_{model_version}.h5")
    models['meta'] = joblib.load(f"{model_dir}/meta_model_{model_version}.joblib")

    return models

def predict_emotion(data, models):
    """
    Dự đoán cảm xúc từ dữ liệu đầu vào sử dụng các mô hình đã tải.

    Args:
        data (np.array): Dữ liệu đầu vào để dự đoán.
        models (dict): Từ điển chứa các mô hình đã tải.

    Returns:
        str: Nhãn cảm xúc dự đoán.
    """
    # Lấy dự đoán từ các mô hình cơ sở
    rf_preds = models['random_forest'].predict_proba(data)
    svm_preds = models['svm'].predict_proba(data)
    gboost_preds = models['gboost'].predict_proba(data)
    nn_preds = models['nn'].predict(data)  # Chú ý đến kích thước và hình dạng của dữ liệu đầu vào

    # Kết hợp dự đoán cho mô hình meta
    meta_data = np.hstack((rf_preds, svm_preds, gboost_preds, nn_preds))

    # Dự đoán cuối cùng từ mô hình meta
    final_prediction = models['meta'].predict(meta_data)
    return final_prediction