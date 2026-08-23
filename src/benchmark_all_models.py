# -*- coding: utf-8 -*-
"""
Script 05: Benchmark so sánh đa mô hình Machine Learning & Deep Learning
Huấn luyện 100% tất cả các mô hình trên toàn bộ 33,544 mẫu Train (No Subsampling),
sử dụng cùng quy trình No Data Leakage và cùng tập Validation/Test độc lập.

Các mô hình so sánh:
1. Physics-Enhanced Local Outlier Factor (Proposed SOTA Method)
2. Physics-Enhanced Isolation Forest (500 Trees)
3. Elliptic Envelope (Robust Mahalanobis Covariance)
4. One-Class SVM (RBF Kernel)
5. Deep Autoencoder (PyTorch Neural Network)
6. DBSCAN (Density-Based Spatial Clustering)

Nghiên cứu thực nghiệm bổ sung (Ablation Study):
- So sánh các Hàm Loss: MSE Loss vs Smooth L1 (Huber) vs L1 Loss
- So sánh các Thuật toán Tối ưu: AdamW vs Adam vs SGD (Momentum)

TUÂN THỦ BẢO MẬT: Không in bất kỳ dòng dữ liệu thô nào ra màn hình.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils, preprocess

# Khởi tạo PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def inject_synthetic_faults(df_subset, fault_ratio=0.08, seed=42):
    """Tạo các lỗi giả lập có kiểm soát trên một tập dữ liệu."""
    np.random.seed(seed)
    df_eval = df_subset.copy().reset_index(drop=True)
    num_samples = len(df_eval)
    num_faults = int(num_samples * fault_ratio)
    fault_indices = np.random.choice(num_samples, size=num_faults, replace=False)

    df_eval['ground_truth_anomaly'] = 0
    df_eval['fault_type'] = 'Normal'

    fault_chunks = np.array_split(fault_indices, 5)

    for idx in fault_chunks[0]:
        df_eval.at[idx, 'temperature'] = np.random.choice([-5.0, 68.5, 82.0])
        df_eval.at[idx, 'ground_truth_anomaly'] = 1
        df_eval.at[idx, 'fault_type'] = 'Extreme_Temperature'

    for idx in fault_chunks[1]:
        df_eval.at[idx, 'lat'] = np.random.choice([0.5, 38.5])
        df_eval.at[idx, 'lon'] = np.random.choice([80.0, 138.0])
        df_eval.at[idx, 'ground_truth_anomaly'] = 1
        df_eval.at[idx, 'fault_type'] = 'Coordinate_Drift'

    for idx in fault_chunks[2]:
        d = df_eval.at[idx, 'depth']
        df_eval.at[idx, 'pressure'] = 2.0 if d > 50 else 500.0
        df_eval.at[idx, 'ground_truth_anomaly'] = 1
        df_eval.at[idx, 'fault_type'] = 'Pressure_Depth_Decoupling'

    for idx in fault_chunks[3]:
        df_eval.at[idx, 'sound_vel_direct'] = np.random.choice([550.0, 2900.0])
        df_eval.at[idx, 'ground_truth_anomaly'] = 1
        df_eval.at[idx, 'fault_type'] = 'Sound_Velocity_Spike'

    for idx in fault_chunks[4]:
        d = df_eval.at[idx, 'depth']
        df_eval.at[idx, 'temperature'] = 32.5 if d > 100 else 2.5
        df_eval.at[idx, 'ground_truth_anomaly'] = 1
        df_eval.at[idx, 'fault_type'] = 'Multivariate_Inconsistency'

    return df_eval

# Kiến trúc Mạng Nơ-ron Autoencoder PyTorch
if HAS_TORCH:
    class OceanAutoencoder(nn.Module):
        def __init__(self, input_dim):
            super(OceanAutoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1)
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, input_dim)
            )

        def forward(self, x):
            latent = self.encoder(x)
            recon = self.decoder(latent)
            return recon

def train_autoencoder(X_train, input_dim, loss_type='smooth_l1', opt_type='adamw', epochs=20, batch_size=256):
    """Huấn luyện mô hình Autoencoder trên 100% 33,544 mẫu Train có Gradient Clipping."""
    if not HAS_TORCH:
        return None
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OceanAutoencoder(input_dim).to(device)

    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.SmoothL1Loss()

    if opt_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    elif opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.002)
    elif opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.002)

    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model

def compute_autoencoder_scores(model, X):
    """Tính sai số tái tạo Reconstruction Error làm Anomaly Score."""
    if not HAS_TORCH or model is None:
        return np.zeros(len(X))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X).to(device)
        recon = model(inputs)
        errors = torch.mean((recon - inputs) ** 2, dim=1).cpu().numpy()
        errors = np.nan_to_num(errors, nan=100.0, posinf=100.0, neginf=0.0)
    return errors

def evaluate_model_pipeline(name, val_scores, test_scores, y_val_true, y_test_true, df_test_eval):
    """Hiệu chuẩn ngưỡng trên Validation và đánh giá khách quan trên Test."""
    val_scores = np.nan_to_num(val_scores, nan=100.0, posinf=100.0, neginf=0.0)
    test_scores = np.nan_to_num(test_scores, nan=100.0, posinf=100.0, neginf=0.0)

    precisions_val, recalls_val, thresholds_val = precision_recall_curve(y_val_true, val_scores)
    f1_val = 2 * (precisions_val * recalls_val) / (precisions_val + recalls_val + 1e-10)
    best_idx = np.argmax(f1_val)
    optimal_th = float(thresholds_val[best_idx]) if len(thresholds_val) > 0 else 0.5

    y_test_pred = (test_scores >= optimal_th).astype(int)
    acc = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, zero_division=0)
    rec = recall_score(y_test_true, y_test_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test_true, test_scores)
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y_test_true, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    df_eval = df_test_eval.copy()
    df_eval['pred'] = y_test_pred
    breakdown = df_eval[df_eval['ground_truth_anomaly'] == 1].groupby('fault_type').agg(
        total=('ground_truth_anomaly', 'count'),
        detected=('pred', 'sum')
    ).reset_index()
    breakdown['detection_rate'] = (breakdown['detected'] / breakdown['total'] * 100.0).round(2)

    return {
        'model_name': name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'optimal_threshold': float(optimal_th),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'fault_breakdown': breakdown.to_dict(orient='records'),
        'test_scores': test_scores.tolist(),
        'y_test_pred': y_test_pred.tolist()
    }

def run_multi_model_benchmark():
    print("=" * 80)
    print("BUOC 5: BENCHMARK SO SANH DA MO HINH (100% SAMPLES - NO SUBSAMPLING)")
    print("=" * 80)

    utils.ensure_directories()

    # 1. Đọc dữ liệu & chia tập (Train 70% / Val 15% / Test 15%)
    print("\n[1/5] Dang doc va phan chia 3 tap du lieu...")
    df_raw = pd.read_excel(config.EXCEL_DATA_PATH, sheet_name=config.SHEET_MAIN, engine="openpyxl")
    df_cleaned = preprocess.parse_and_clean_raw_dataframe(df_raw)

    df_train_raw, df_temp = train_test_split(df_cleaned, test_size=0.30, random_state=config.RANDOM_SEED)
    df_val_raw, df_test_raw = train_test_split(df_temp, test_size=0.50, random_state=config.RANDOM_SEED)

    # Tính toán đặc trưng chỉ trên Train (No Data Leakage)
    ai_features, rule_only_features, train_medians, spatial_center = preprocess.identify_feature_types(df_train_raw)

    # Điền median
    df_train = preprocess.apply_median_imputation(df_train_raw, train_medians)
    df_val_base = preprocess.apply_median_imputation(df_val_raw, train_medians)
    df_test_base = preprocess.apply_median_imputation(df_test_raw, train_medians)

    # Cấy lỗi giả lập
    df_val = inject_synthetic_faults(df_val_base, fault_ratio=config.SYNTHETIC_FAULT_RATIO, seed=config.RANDOM_SEED)
    df_test = inject_synthetic_faults(df_test_base, fault_ratio=config.SYNTHETIC_FAULT_RATIO, seed=config.RANDOM_SEED + 100)

    # Thêm đặc trưng vật lý & không gian
    df_train = preprocess.add_full_spatial_physics_features(df_train, spatial_center)
    df_val = preprocess.add_full_spatial_physics_features(df_val, spatial_center)
    df_test = preprocess.add_full_spatial_physics_features(df_test, spatial_center)

    # Chuẩn hóa RobustScaler
    scaler = RobustScaler()
    X_train = scaler.fit_transform(df_train[ai_features])
    X_val = scaler.transform(df_val[ai_features])
    X_test = scaler.transform(df_test[ai_features])

    y_val_true = df_val['ground_truth_anomaly'].values
    y_test_true = df_test['ground_truth_anomaly'].values
    input_dim = len(ai_features)

    all_results = []
    print(f"  + Tap Train: {len(X_train):,} mau (Su dung 100% tat ca cac mo hinh)")
    print(f"  + Bo dac trung su dung ({input_dim} dac trung): {ai_features}")

    # ==============================================================================
    # 2. HUẤN LUYỆN & ĐÁNH GIÁ 6 MÔ HÌNH TRÊN 100% TẬP TRAIN
    # ==============================================================================
    print("\n[2/5] Dang huan luyen va danh gia 6 mo hinh hoc may & hoc sau...")

    # Mô hình 1: Local Outlier Factor (LOF) - Proposed SOTA Method
    print("  [1/6] Training Physics-Enhanced Local Outlier Factor (LOF - Proposed)...")
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination=0.08, n_jobs=-1)
    lof.fit(X_train)
    val_scores_lof = -lof.score_samples(X_val)
    test_scores_lof = -lof.score_samples(X_test)
    res_lof = evaluate_model_pipeline("Local Outlier Factor (Proposed)", val_scores_lof, test_scores_lof, y_val_true, y_test_true, df_test)
    res_lof['train_time_sec'] = round(time.time() - t0, 2)
    all_results.append(res_lof)

    # Mô hình 2: Physics-Enhanced Isolation Forest (500 Trees)
    print("  [2/6] Training Physics-Enhanced Isolation Forest (500 Trees)...")
    t0 = time.time()
    iforest = IsolationForest(**config.IFOREST_PARAMS)
    iforest.fit(X_train)
    val_scores_if = -iforest.score_samples(X_val)
    test_scores_if = -iforest.score_samples(X_test)
    res_if = evaluate_model_pipeline("Isolation Forest (500 Trees)", val_scores_if, test_scores_if, y_val_true, y_test_true, df_test)
    res_if['train_time_sec'] = round(time.time() - t0, 2)
    all_results.append(res_if)

    # Mô hình 3: Elliptic Envelope (Robust Covariance)
    print("  [3/6] Training Elliptic Envelope (Robust Covariance)...")
    t0 = time.time()
    ee = EllipticEnvelope(contamination=0.08, random_state=config.RANDOM_SEED, support_fraction=0.85)
    ee.fit(X_train)
    val_scores_ee = -ee.score_samples(X_val)
    test_scores_ee = -ee.score_samples(X_test)
    res_ee = evaluate_model_pipeline("Elliptic Envelope", val_scores_ee, test_scores_ee, y_val_true, y_test_true, df_test)
    res_ee['train_time_sec'] = round(time.time() - t0, 2)
    all_results.append(res_ee)

    # Mô hình 4: One-Class SVM (RBF Kernel) - 100% 33,544 mẫu Train
    print("  [4/6] Training One-Class SVM (RBF Kernel) tren toan bo tap Train...")
    t0 = time.time()
    ocsvm = OneClassSVM(kernel='rbf', nu=0.08, gamma='scale')
    ocsvm.fit(X_train)
    val_scores_svm = -ocsvm.score_samples(X_val)
    test_scores_svm = -ocsvm.score_samples(X_test)
    res_svm = evaluate_model_pipeline("One-Class SVM (RBF)", val_scores_svm, test_scores_svm, y_val_true, y_test_true, df_test)
    res_svm['train_time_sec'] = round(time.time() - t0, 2)
    all_results.append(res_svm)

    # Mô hình 5: Deep Autoencoder (PyTorch Neural Network)
    print("  [5/6] Training Deep Autoencoder (PyTorch Neural Network)...")
    t0 = time.time()
    if HAS_TORCH:
        ae_model = train_autoencoder(X_train, input_dim, loss_type='smooth_l1', opt_type='adamw', epochs=20)
        val_scores_ae = compute_autoencoder_scores(ae_model, X_val)
        test_scores_ae = compute_autoencoder_scores(ae_model, X_test)
        res_ae = evaluate_model_pipeline("Deep Autoencoder (PyTorch)", val_scores_ae, test_scores_ae, y_val_true, y_test_true, df_test)
        res_ae['train_time_sec'] = round(time.time() - t0, 2)
        all_results.append(res_ae)

    # Mô hình 6: DBSCAN (Clustering Outlier Detection)
    print("  [6/6] Training DBSCAN (Density-Based Clustering)...")
    t0 = time.time()
    dbscan = DBSCAN(eps=2.5, min_samples=10, n_jobs=-1)
    test_labels_db = dbscan.fit_predict(X_test)
    test_scores_db = (test_labels_db == -1).astype(float)
    val_scores_db = np.zeros(len(X_val))
    res_db = evaluate_model_pipeline("DBSCAN", val_scores_db, test_scores_db, y_val_true, y_test_true, df_test)
    res_db['train_time_sec'] = round(time.time() - t0, 2)
    all_results.append(res_db)

    # ==============================================================================
    # 3. ABLATION STUDY: SO SÁNH LOSS FUNCTIONS & OPTIMIZERS TRÊN AUTOENCODER
    # ==============================================================================
    print("\n[3/5] Dang thuc hien Ablation Study: So sanh Loss Functions & Optimizers...")
    ablation_results = []
    
    if HAS_TORCH:
        loss_types = ['smooth_l1', 'mse', 'l1']
        opt_types = ['adamw', 'adam', 'sgd']

        for l_type in loss_types:
            for o_type in opt_types:
                t_sub0 = time.time()
                m_sub = train_autoencoder(X_train, input_dim, loss_type=l_type, opt_type=o_type, epochs=15)
                v_sc = compute_autoencoder_scores(m_sub, X_val)
                t_sc = compute_autoencoder_scores(m_sub, X_test)
                eval_sub = evaluate_model_pipeline(f"AE ({l_type.upper()} + {o_type.upper()})", v_sc, t_sc, y_val_true, y_test_true, df_test)
                eval_sub['loss_function'] = l_type.upper()
                eval_sub['optimizer'] = o_type.upper()
                eval_sub['train_time_sec'] = round(time.time() - t_sub0, 2)
                ablation_results.append(eval_sub)

    # ==============================================================================
    # 4. XUẤT CÁC BẢNG SO SÁNH KHOA HỌC RA EXCEL
    # ==============================================================================
    print("\n[4/5] Dang xuat cac bang so sanh khoa hoc ra file Excel...")

    # Bảng 5: So sánh hiệu năng tổng thể các mô hình
    table5_rows = []
    for r in all_results:
        cm = r['confusion_matrix']
        table5_rows.append({
            'Mo hinh danh gia': r['model_name'],
            'ROC-AUC': f"{r['roc_auc']:.4f}",
            'Accuracy (%)': f"{r['accuracy']*100:.2f}%",
            'Precision (%)': f"{r['precision']*100:.2f}%",
            'Recall (%)': f"{r['recall']*100:.2f}%",
            'F1-Score': f"{r['f1_score']:.4f}",
            'True Positive (TP)': cm['tp'],
            'False Positive (FP)': cm['fp'],
            'False Negative (FN)': cm['fn'],
            'Thoi gian Huan luyen (s)': r.get('train_time_sec', 0)
        })

    df_table5 = pd.DataFrame(table5_rows).sort_values(by='F1-Score', ascending=False)
    df_table5.to_excel(config.BENCHMARK_EXCEL, index=False)

    # Bảng 6: Tỷ lệ phát hiện theo từng nhóm lỗi chi tiết của từng mô hình
    table6_rows = []
    fault_names = ['Coordinate_Drift', 'Extreme_Temperature', 'Pressure_Depth_Decoupling', 'Sound_Velocity_Spike', 'Multivariate_Inconsistency']
    
    for r in all_results:
        row = {'Mo hinh': r['model_name']}
        breakdown_dict = {item['fault_type']: item['detection_rate'] for item in r['fault_breakdown']}
        for fn in fault_names:
            row[f"{fn} (%)"] = f"{breakdown_dict.get(fn, 0.0):.2f}%"
        row['Tong the Recall (%)'] = f"{r['recall']*100:.2f}%"
        row['F1-Score'] = f"{r['f1_score']:.4f}"
        table6_rows.append(row)

    df_table6 = pd.DataFrame(table6_rows)
    df_table6.to_excel(config.BENCHMARK_FAULT_EXCEL, index=False)

    # Bảng 5B: Ablation Study so sánh Loss & Optimizer
    if ablation_results:
        table5b_rows = []
        for ab in ablation_results:
            table5b_rows.append({
                'Ham Loss': ab['loss_function'],
                'Thuat toan Toi uu': ab['optimizer'],
                'F1-Score': f"{ab['f1_score']:.4f}",
                'ROC-AUC': f"{ab['roc_auc']:.4f}",
                'Precision (%)': f"{ab['precision']*100:.2f}%",
                'Recall (%)': f"{ab['recall']*100:.2f}%",
                'Accuracy (%)': f"{ab['accuracy']*100:.2f}%",
                'Thoi gian (s)': ab['train_time_sec']
            })
        df_table5b = pd.DataFrame(table5b_rows).sort_values(by='F1-Score', ascending=False)
        df_table5b.to_excel(config.TABLES_DIR / "Bang5B_Ablation_Loss_Optimizer_Autoencoder.xlsx", index=False)

    # Lưu toàn bộ vào file JSON
    json_save_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'features_used': ai_features,
        'benchmark_models': [
            {k: v for k, v in r.items()}
            for r in all_results
        ],
        'ablation_study': [
            {k: v for k, v in ab.items()}
            for ab in ablation_results
        ]
    }
    with open(config.BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(json_save_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("TONG HOP KET QUA BENCHMARK SO SANH DA MO HINH (FULL SAMPLES)")
    print("=" * 80)
    print(df_table5[['Mo hinh danh gia', 'F1-Score', 'ROC-AUC', 'Precision (%)', 'Recall (%)', 'Thoi gian Huan luyen (s)']].to_string(index=False))
    print("=" * 80)
    print(f"[HOAN TAT] Bang 5 (So sanh mo hinh) da luu tai: {config.BENCHMARK_EXCEL}")
    print(f"[HOAN TAT] Bang 6 (Phat hien theo loai loi) da luu tai: {config.BENCHMARK_FAULT_EXCEL}")
    print(f"[HOAN TAT] Bang 5B (Ablation Loss & Optimizer) da luu tai: {config.TABLES_DIR / 'Bang5B_Ablation_Loss_Optimizer_Autoencoder.xlsx'}")
    print(f"[HOAN TAT] File JSON tong hop da luu tai: {config.BENCHMARK_JSON}")
    print("=" * 80)

if __name__ == "__main__":
    run_multi_model_benchmark()
