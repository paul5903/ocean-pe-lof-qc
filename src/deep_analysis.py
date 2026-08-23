# -*- coding: utf-8 -*-
"""
Script 06: Phân tích sâu dữ liệu thực địa & Xuất bộ bảng biểu, biểu đồ khoa học toàn diện (300 DPI)
cho bài báo khoa học về Kiểm soát chất lượng dữ liệu hải dương học (Ocean Data QC).

Các sản phẩm tạo ra:
- Bảng 7: Thống kê mô tả dữ liệu thực địa (Min, Max, Mean, Median, Std, Q1, Q3, Missing%, Skewness, Kurtosis)
- Bảng 8: Ma trận tương quan Pearson giữa các biến hải dương học
- Bảng 9: Thống kê phân bố dữ liệu theo các tầng nước (0-50m, 50-200m, 200-1000m, >1000m)
- Bảng 10: Thống kê phân bố dữ liệu theo các phân vùng biển (Vịnh Bắc Bộ, Miền Trung, Nam Bộ)
- Hình 5: Biểu đồ so sánh hiệu năng tổng thể các mô hình (F1, Precision, Recall, Accuracy)
- Hình 6: Đường cong ROC đa mô hình (Multi-model ROC Curves)
- Hình 7: Đường cong Precision-Recall đa mô hình (Multi-model PR Curves)
- Hình 8: Biểu đồ Radar đa chiều (Spider Chart) so sánh các mô hình
- Hình 9: Grid Boxplots phân bố các biến đo đạc thực địa
- Hình 10: Heatmap ma trận tương quan Pearson chuẩn bài báo
- Hình 11: Histogram & KDE phân bố Nhiệt độ theo tầng nước
- Hình 12: Biểu đồ tỷ lệ phát hiện theo 5 nhóm lỗi của các mô hình

TUÂN THỦ BẢO MẬT: Không in bất kỳ dòng dữ liệu thô nào ra màn hình.
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils, preprocess

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

def generate_deep_analysis_and_visuals():
    print("=" * 80)
    print("BUOC 6: PHAN TICH SAU DU LIEU THUC DIA & TAO BO BIEU DO KHOA HOC (300 DPI)")
    print("=" * 80)

    utils.ensure_directories()
    utils.setup_plot_style()

    # 1. Đọc dữ liệu thô và làm sạch cơ bản
    print("\n[1/6] Dang tai va tien xu ly du lieu tu Sheet 1...")
    df_raw = pd.read_excel(config.EXCEL_DATA_PATH, sheet_name=config.SHEET_MAIN, engine="openpyxl")
    df = preprocess.parse_and_clean_raw_dataframe(df_raw)
    total_records = len(df)

    # ==============================================================================
    # 2. XUẤT BẢNG 7: THỐNG KÊ MÔ TẢ DỮ LIỆU THỰC ĐỊA TOÀN DIỆN
    # ==============================================================================
    print("\n[2/6] Dang tinh toan Bang 7: Thong ke mo ta du lieu thuc dia...")
    
    target_cols = [
        ('lat', 'Vĩ độ (°N)'),
        ('lon', 'Kinh độ (°E)'),
        ('depth', 'Tầng nước / Độ sâu (m)'),
        ('temperature', 'Nhiệt độ nước biển (°C)'),
        ('pressure', 'Áp suất thủy tĩnh (dbar)'),
        ('sound_vel_direct', 'Vận tốc âm trực tiếp (m/s)'),
        ('salinity', 'Độ muối (‰)'),
        ('density', 'Tỷ trọng (kg/m³)'),
        ('conductivity', 'Độ dẫn điện (mS/cm)'),
        ('turbidity', 'Độ đục (mg/L)'),
        ('sound_vel_calc', 'Vận tốc âm tính toán (m/s)')
    ]

    desc_rows = []
    for col_key, col_label in target_cols:
        if col_key in df.columns:
            series = df[col_key].dropna()
            valid_cnt = len(series)
            missing_cnt = total_records - valid_cnt
            missing_pct = (missing_cnt / total_records) * 100.0

            if valid_cnt > 0:
                mean_val = float(series.mean())
                std_val = float(series.std())
                min_val = float(series.min())
                q1_val = float(series.quantile(0.25))
                med_val = float(series.median())
                q3_val = float(series.quantile(0.75))
                max_val = float(series.max())
                iqr_val = q3_val - q1_val
                skew_val = float(stats.skew(series))
                kurt_val = float(stats.kurtosis(series))
            else:
                mean_val = std_val = min_val = q1_val = med_val = q3_val = max_val = iqr_val = skew_val = kurt_val = 0.0

            desc_rows.append({
                'STT': len(desc_rows) + 1,
                'Tham số đo đạc': col_label,
                'Số mẫu hợp lệ': f"{valid_cnt:,}",
                'Tỷ lệ khuyết (%)': f"{missing_pct:.2f}%",
                'Giá trị Min': f"{min_val:.2f}",
                'Phân vị Q1 (25%)': f"{q1_val:.2f}",
                'Trung vị (Median)': f"{med_val:.2f}",
                'Phân vị Q3 (75%)': f"{q3_val:.2f}",
                'Giá trị Max': f"{max_val:.2f}",
                'Giá trị Mean': f"{mean_val:.2f}",
                'Độ lệch chuẩn (Std)': f"{std_val:.2f}",
                'Độ trải giữa (IQR)': f"{iqr_val:.2f}",
                'Độ lệch (Skewness)': f"{skew_val:.2f}",
                'Độ nhọn (Kurtosis)': f"{kurt_val:.2f}"
            })

    df_desc = pd.DataFrame(desc_rows)
    df_desc.to_excel(config.DESCRIPTIVE_STATS_EXCEL, index=False)
    print(f"  + Da luu Bang 7: {config.DESCRIPTIVE_STATS_EXCEL}")

    # ==============================================================================
    # 3. XUẤT BẢNG 8, 9, 10: TƯƠNG QUAN & PHÂN BỐ TẦNG NƯỚC, VÙNG BIỂN
    # ==============================================================================
    print("\n[3/6] Dang tao Bang 8, 9, 10 (Tuong quan, Tang nuoc, Vung bien)...")

    # Bảng 8: Ma trận tương quan Pearson
    numeric_sub_cols = ['lat', 'lon', 'depth', 'temperature', 'pressure', 'sound_vel_direct', 'salinity', 'density']
    available_num_cols = [c for c in numeric_sub_cols if c in df.columns]
    corr_matrix = df[available_num_cols].corr(method='pearson')
    corr_matrix.to_excel(config.CORRELATION_EXCEL)

    # Bảng 9: Phân bố dữ liệu theo các tầng nước
    depth_bins = [0, 50, 200, 1000, 6000]
    depth_labels = [
        'Tầng mặt & bề mặt (0 - 50m)',
        'Tầng dị nhiệt / dưới mặt (50 - 200m)',
        'Tầng trung gian (200 - 1000m)',
        'Tầng biển sâu (> 1000m)'
    ]
    df['depth_zone'] = pd.cut(df['depth'], bins=depth_bins, labels=depth_labels, right=True)
    depth_summary = df['depth_zone'].value_counts(dropna=False).reset_index()
    depth_summary.columns = ['Phân tầng độ sâu', 'Số lượng bản ghi']
    depth_summary['Tỷ lệ (%)'] = (depth_summary['Số lượng bản ghi'] / total_records * 100.0).round(2)
    depth_summary.to_excel(config.DEPTH_ZONE_EXCEL, index=False)

    # Bảng 10: Phân bố dữ liệu theo các phân vùng biển
    def classify_region(lat):
        if pd.isna(lat):
            return 'Chưa xác định'
        if lat >= 19.0:
            return 'Vùng biển Vịnh Bắc Bộ (>= 19°N)'
        elif lat >= 12.0:
            return 'Vùng biển Miền Trung (12°N - 19°N)'
        else:
            return 'Vùng biển Nam Bộ & Tây Nam Bộ (< 12°N)'

    df['region_zone'] = df['lat'].apply(classify_region)
    region_summary = df['region_zone'].value_counts().reset_index()
    region_summary.columns = ['Khu vực biển', 'Số lượng bản ghi']
    region_summary['Tỷ lệ (%)'] = (region_summary['Số lượng bản ghi'] / total_records * 100.0).round(2)
    region_summary.to_excel(config.REGION_EXCEL, index=False)

    print(f"  + Da luu Bang 8, 9, 10 vao thu muc: {config.TABLES_DIR}")

    # ==============================================================================
    # 4. VẼ CÁC BIỂU ĐỒ KHOA HỌC 300 DPI CHO BÀI BÁO (HÌNH 5 ĐẾN HÌNH 12)
    # ==============================================================================
    print("\n[4/6] Dang ve cac bieu do khoa hoc mo rong 300 DPI (Hinh 5 - Hinh 12)...")

    # Đọc kết quả benchmark
    has_benchmark = config.BENCHMARK_JSON.exists()
    if has_benchmark:
        with open(config.BENCHMARK_JSON, "r", encoding="utf-8") as f:
            bench_data = json.load(f)
        models_list = bench_data.get('benchmark_models', [])
    else:
        models_list = []

    # HÌNH 5: Biểu đồ cột so sánh hiệu năng giữa các mô hình (Bar Chart)
    if models_list:
        fig5_path = config.FIGURES_DIR / "Hinh5_So_Sanh_Hieu_Nang_Mo_Hinh.png"
        df_bm = pd.DataFrame(models_list)
        df_bm['F1_pct'] = df_bm['f1_score'] * 100.0
        df_bm['Prec_pct'] = df_bm['precision'] * 100.0
        df_bm['Rec_pct'] = df_bm['recall'] * 100.0
        df_bm['Acc_pct'] = df_bm['accuracy'] * 100.0

        x = np.arange(len(df_bm))
        width = 0.20

        plt.figure(figsize=(12, 6))
        plt.bar(x - 1.5*width, df_bm['F1_pct'], width, label='F1-Score (%)', color='#1f77b4', edgecolor='black')
        plt.bar(x - 0.5*width, df_bm['Prec_pct'], width, label='Precision (%)', color='#2ca02c', edgecolor='black')
        plt.bar(x + 0.5*width, df_bm['Rec_pct'], width, label='Recall (%)', color='#ff7f0e', edgecolor='black')
        plt.bar(x + 1.5*width, df_bm['Acc_pct'], width, label='Accuracy (%)', color='#9467bd', edgecolor='black')

        plt.ylabel('Score (%)', fontsize=12)
        plt.title('Performance Comparison of Machine Learning & Deep Learning Models for Ocean QC', fontsize=13, pad=14)
        plt.xticks(x, df_bm['model_name'], rotation=15, ha='right', fontsize=10)
        plt.ylim(0, 110)
        plt.legend(loc='lower right', frameon=True)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig5_path, dpi=300)
        plt.close()
        print(f"  + [Hinh 5]: {fig5_path.name}")

    # HÌNH 6 & HÌNH 7: Đường cong ROC & Precision-Recall Đa mô hình
    if models_list and 'test_scores' in models_list[0]:
        # Tái tạo ground truth trên tập Test
        df_cleaned = preprocess.parse_and_clean_raw_dataframe(df_raw)
        _, df_temp = train_test_split(df_cleaned, test_size=0.30, random_state=config.RANDOM_SEED)
        _, df_test_raw = train_test_split(df_temp, test_size=0.50, random_state=config.RANDOM_SEED)
        
        df_test_eval = inject_synthetic_faults(df_test_raw, fault_ratio=config.SYNTHETIC_FAULT_RATIO, seed=config.RANDOM_SEED + 100)
        y_test_gt = df_test_eval['ground_truth_anomaly'].values

        # HÌNH 6: Đường cong ROC đa mô hình
        fig6_path = config.FIGURES_DIR / "Hinh6_Duong_Cong_ROC_Da_Mo_Hinh.png"
        plt.figure(figsize=(8, 7))
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for idx, m in enumerate(models_list):
            if 'test_scores' in m:
                scores = np.array(m['test_scores'])
                fpr, tpr, _ = roc_curve(y_test_gt, scores)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color_palette[idx % len(color_palette)], lw=2, label=f"{m['model_name']} (AUC = {roc_auc:.4f})")

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
        plt.title('Multi-Model Receiver Operating Characteristic (ROC) Curves', fontsize=13, pad=12)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig6_path, dpi=300)
        plt.close()
        print(f"  + [Hinh 6]: {fig6_path.name}")

        # HÌNH 7: Đường cong Precision-Recall đa mô hình
        fig7_path = config.FIGURES_DIR / "Hinh7_Duong_Cong_Precision_Recall_Da_Mo_Hinh.png"
        plt.figure(figsize=(8, 7))

        for idx, m in enumerate(models_list):
            if 'test_scores' in m:
                scores = np.array(m['test_scores'])
                pr_p, pr_r, _ = precision_recall_curve(y_test_gt, scores)
                pr_auc = auc(pr_r, pr_p)
                plt.plot(pr_r, pr_p, color=color_palette[idx % len(color_palette)], lw=2, label=f"{m['model_name']} (PR-AUC = {pr_auc:.4f})")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Detection Rate)', fontsize=11)
        plt.ylabel('Precision (Positive Predictive Value)', fontsize=11)
        plt.title('Multi-Model Precision-Recall Curves across Thresholds', fontsize=13, pad=12)
        plt.legend(loc="lower left", fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig7_path, dpi=300)
        plt.close()
        print(f"  + [Hinh 7]: {fig7_path.name}")

    # HÌNH 8: Biểu đồ Radar đa chiều (Spider Chart)
    if models_list:
        fig8_path = config.FIGURES_DIR / "Hinh8_Radar_Chart_So_Sanh_Da_Chieu.png"
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories, size=11)
        ax.set_rlabel_position(0)
        plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], ["0.6", "0.7", "0.8", "0.9", "1.0"], color="grey", size=9)
        plt.ylim(0.0, 1.05)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for idx, m in enumerate(models_list):
            values = [m['accuracy'], m['precision'], m['recall'], m['f1_score'], m['roc_auc']]
            values += values[:1]
            c = colors[idx % len(colors)]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=m['model_name'], color=c)
            ax.fill(angles, values, color=c, alpha=0.1)

        plt.title("Multi-Dimensional Performance Radar Chart across Models", size=13, y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
        plt.tight_layout()
        plt.savefig(fig8_path, dpi=300)
        plt.close()
        print(f"  + [Hinh 8]: {fig8_path.name}")

    # HÌNH 9: Grid Boxplots phân bố các biến đo đạc thực địa
    fig9_path = config.FIGURES_DIR / "Hinh9_Boxplot_Phan_Bo_Cac_Bien_Do.png"
    plot_vars = ['depth', 'temperature', 'pressure', 'sound_vel_direct']
    var_titles = ['Depth (m)', 'Temperature (°C)', 'Pressure (dbar)', 'Sound Velocity (m/s)']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for i, var in enumerate(plot_vars):
        if var in df.columns:
            sns.boxplot(y=df[var].dropna(), ax=axes[i], color='#3498db', width=0.4, fliersize=2)
            axes[i].set_title(var_titles[i], fontsize=12, pad=10)
            axes[i].set_ylabel('')
            axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle('Distribution and Natural Dispersion of In-situ Oceanographic Measurements', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print(f"  + [Hinh 9]: {fig9_path.name}")

    # HÌNH 10: Ma trận tương quan Pearson (Heatmap)
    fig10_path = config.FIGURES_DIR / "Hinh10_Ma_Tran_Tuong_Quan_Heatmap.png"
    plt.figure(figsize=(9, 7))
    rename_dict = {
        'lat': 'Lat', 'lon': 'Lon', 'depth': 'Depth', 'temperature': 'Temp',
        'pressure': 'Pres', 'sound_vel_direct': 'SoundVel', 'salinity': 'Sal', 'density': 'Density'
    }
    sub_corr = corr_matrix.rename(index=rename_dict, columns=rename_dict)
    sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0, cbar_kws={'label': 'Pearson Correlation (r)'})
    plt.title('Inter-variable Pearson Correlation Matrix of Marine Parameters', fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    print(f"  + [Hinh 10]: {fig10_path.name}")

    # HÌNH 11: Histogram & KDE phân bố Nhiệt độ theo tầng nước
    fig11_path = config.FIGURES_DIR / "Hinh11_Histogram_KDE_NhietDo_DoSau.png"
    plt.figure(figsize=(10, 5.5))
    valid_td = df.dropna(subset=['temperature', 'depth_zone'])
    if len(valid_td) > 0:
        sns.histplot(data=valid_td, x='temperature', hue='depth_zone', bins=40, kde=True, element="step", common_norm=False, palette="Set1")
        plt.title('Thermal Distribution (KDE) Stratified by Ocean Depth Zones', fontsize=13, pad=12)
        plt.xlabel('Temperature (°C)', fontsize=11)
        plt.ylabel('Frequency Density', fontsize=11)
        plt.tight_layout()
        plt.savefig(fig11_path, dpi=300)
        plt.close()
        print(f"  + [Hinh 11]: {fig11_path.name}")

    # HÌNH 12: Biểu đồ tỷ lệ phát hiện theo từng nhóm lỗi của các mô hình
    if models_list:
        fig12_path = config.FIGURES_DIR / "Hinh12_So_Sanh_Phat_Hien_Theo_Nhom_Loi.png"
        fault_data = []
        for m in models_list:
            for fb in m.get('fault_breakdown', []):
                fault_data.append({
                    'Model': m['model_name'],
                    'Fault_Type': fb['fault_type'],
                    'Detection_Rate': fb['detection_rate']
                })
        
        if fault_data:
            df_fd = pd.DataFrame(fault_data)
            plt.figure(figsize=(14, 6))
            sns.barplot(data=df_fd, x='Fault_Type', y='Detection_Rate', hue='Model', palette='tab10', edgecolor='black')
            plt.title('Anomaly Detection Rate by Fault Category across Diverse Machine Learning Models', fontsize=13, pad=12)
            plt.ylabel('Detection Rate (%)', fontsize=11)
            plt.xlabel('Fault Categories', fontsize=11)
            plt.ylim(0, 110)
            plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.32), ncol=3, frameon=True, fontsize=9)
            plt.xticks(rotation=10, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(fig12_path, dpi=300)
            plt.close()
            print(f"  + [Hinh 12]: {fig12_path.name}")

    print("\n" + "=" * 80)
    print(f"[HOAN TAT] Toan bo bang so lieu (Bang 7 - 10) tai: {config.TABLES_DIR}")
    print(f"[HOAN TAT] Toan bo bieu do khoa hoc (Hinh 5 - 12) tai: {config.FIGURES_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    generate_deep_analysis_and_visuals()
