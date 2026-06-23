import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import pandas as pd
import numpy as np
import json
import fitz
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG & LANGUAGE
# ─────────────────────────────────────────────────────────────────
# Trigger reload for new datasets
st.set_page_config(
    page_title="ScoreVision",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

lang_choice = st.sidebar.radio("🌐 Language / Ngôn ngữ", ["Tiếng Việt", "English"])
lang = "vi" if lang_choice == "Tiếng Việt" else "en"

def t(vi, en):
    return vi if lang == "vi" else en

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(229,57,53,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(255,255,255,0.65);
    font-size: 0.95rem;
    margin: 0;
}
.hero .badge {
    display: inline-block;
    background: rgba(229,57,53,0.2);
    border: 1px solid rgba(229,57,53,0.4);
    color: #ef9a9a;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #f8f9fc;
    border: 1px solid #e8ecf4;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #8892a4;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1a1a2e;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #adb5bd;
    margin-top: 4px;
}

/* Prediction result */
.pred-box {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    color: white;
}
.pred-box .score {
    font-size: 3.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 8px;
}
.pred-box .conf {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
}
.pred-box .conf span {
    color: #81c784;
    font-weight: 600;
}

/* Top-3 bars */
.bar-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 10px;
}
.bar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    width: 36px;
    color: #1a1a2e;
}
.bar-track {
    flex: 1;
    background: #f0f2f8;
    border-radius: 6px;
    height: 22px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #1a1a2e, #0f3460);
    transition: width 0.6s ease;
}
.bar-pct {
    font-size: 0.8rem;
    color: #8892a4;
    width: 46px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f0f2f8;
    padding: 5px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1);
}

/* Info box */
.info-box {
    background: #e8f4fd;
    border-left: 4px solid #2196f3;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.88rem;
    color: #1565c0;
}

/* Status pill */
.pill-correct {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
}
.pill-wrong {
    display: inline-block;
    background: #ffebee;
    color: #c62828;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
}

/* Section title */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #f0f2f8;
}

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    color: #adb5bd;
    font-size: 0.8rem;
    margin-top: 40px;
    border-top: 1px solid #f0f2f8;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────
BASE   = Path(r"C:\Users\Tung\Downloads\Đồ án chuyên ngành")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCORES = sorted(
    [f"{i}.{d}" for i in range(0, 10) for d in [0, 3, 5, 8]] + ["10"],
    key=lambda x: float(x)
)
idx2label = {i: s for i, s in enumerate(SCORES)}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 41)
    
    # Check if model exists, else just return uninitialized model (for demo purposes if not present)
    model_path = BASE / "models" / "EfficientNet_B0.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    model.to(DEVICE).eval()
    return model

@st.cache_data
def load_template():
    boxes_path = BASE / "Test_Phieu_01" / "phase3b_boxes_manual_full.json"
    img_path = BASE / "Test_Phieu_01" / "page_01.png"
    
    if boxes_path.exists() and img_path.exists():
        boxes = json.load(open(boxes_path))
        ref_img = Image.open(img_path)
        return boxes, ref_img.size
    return {}, (1000, 1400) # dummy fallback

model        = load_model()
boxes_tmpl, ref_size = load_template()

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def predict(img: Image.Image):
    tensor = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out  = model(tensor)
        prob = torch.softmax(out, dim=1)[0]
    top3_idx  = prob.topk(3).indices.tolist()
    top3_prob = prob.topk(3).values.tolist()
    return (
        idx2label[top3_idx[0]],
        top3_prob[0],
        [(idx2label[i], p) for i, p in zip(top3_idx, top3_prob)]
    )

def pdf_to_images(pdf_bytes, dpi=200):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages

def crop_cell(page_img, box, sx, sy):
    return page_img.crop((
        int(box["x1"] * sx), int(box["y1"] * sy),
        int(box["x2"] * sx), int(box["y2"] * sy)
    ))

def compute_gradcam(img: Image.Image):
    model.zero_grad()
    tensor = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)

    activations, gradients = [], []
    h1 = model.features[-1].register_forward_hook(
        lambda m, i, o: activations.append(o.detach()))
    h2 = model.features[-1].register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach()))

    out = model(tensor)
    pred_idx = out.argmax(1).item()
    out[0, pred_idx].backward()
    h1.remove(); h2.remove()

    weights = gradients[0].mean(dim=[2, 3], keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(dim=1)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.array(Image.fromarray(cam).resize(img.size, Image.BILINEAR))
    heatmap = cm.jet(cam)[:, :, :3]
    img_arr  = np.array(img.convert("RGB")) / 255.0
    overlay  = np.clip(0.5 * img_arr + 0.5 * heatmap, 0, 1)
    return (overlay * 255).astype(np.uint8), idx2label[pred_idx]

def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="badge">🎓 {t('Đồ án chuyên ngành', 'Specialized Project')}</div>
    <h1>{t('ScoreVision – Nhận diện điểm số viết tay', 'ScoreVision – Handwritten Score Recognition')}</h1>
    <p>{t('Ứng dụng học sâu nhận diện điểm số viết tay trên phiếu điểm', 'Deep learning app for handwritten score recognition on grading sheets')} &nbsp;·&nbsp;
       {t('Mô hình EfficientNet-B0', 'EfficientNet-B0 Model')} &nbsp;·&nbsp;
       41 classes &nbsp;·&nbsp;
       {t('Accuracy thực tế 77.3%', 'Real-world Accuracy 77.3%')}</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    t("📊  Demo hoàn chỉnh – Phiếu 01", "📊  Full Demo – Sheet 01"),
    t("🔍  Dự đoán thực tế – Upload PDF", "🔍  Real Prediction – Upload PDF"),
    t("🖼️  Dự đoán đơn lẻ", "🖼️  Single Cell Prediction")
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 – DEMO HOÀN CHỈNH
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="section-title">{t("Phiếu điểm mẫu – Kết quả nhận diện đầy đủ", "Sample Grading Sheet – Full Recognition Results")}</div>',
                unsafe_allow_html=True)

    csv_print_path = BASE / "Test_Phieu_01" / "grade_sheet_printed_text_final.csv"
    csv_labels_path = BASE / "Test_Phieu_01" / "external_test_labels.csv"

    if csv_print_path.exists() and csv_labels_path.exists():
        # Load data
        df_print  = pd.read_csv(csv_print_path)
        
        # Determine column names based on language
        col_stt = t("stt", "no.")
        col_sv = t("ma_sv", "student_id")
        col_name = t("ho_ten", "full_name")
        col_dob = t("ng_sinh", "dob")
        col_class = t("ten_lop", "class_name")
        
        col_pred = t("Đ.số dự đoán", "Pred Score")
        col_gt = "Ground Truth"
        col_check = t("Đúng/Sai", "Correct/Wrong")
        
        df_print.columns = [col_stt, col_sv, col_name, col_dob, col_class]
        df_labels = pd.read_csv(csv_labels_path)

        # Predict
        preds, confs = [], []
        for _, row in df_labels.iterrows():
            img_path = row["image_path"]
            if Path(img_path).exists():
                img = Image.open(img_path)
                pred, conf, _ = predict(img)
            else:
                pred, conf = "0.0", 0.0
            preds.append(pred)
            confs.append(conf)

        df_labels["d_so_pred"] = preds
        df_labels["confidence"] = confs
        df_labels["label_str"]  = df_labels["label"].apply(
            lambda x: "10" if str(x) == "10.0" else str(x))

        # Merge
        df_result = df_print.copy()
        df_result[col_pred] = df_labels["d_so_pred"].values
        df_result["Confidence"]   = df_labels["confidence"].round(3).values
        df_result[col_gt] = df_labels["label_str"].values
        df_result[col_check] = df_result.apply(
            lambda r: "⚠️" if str(r[col_gt]) == "nan"
                      else ("✅" if str(r[col_gt]) == str(r[col_pred]) else "❌"),
            axis=1)
        
        # Ô vắng thi → hiển thị NaN thay vì dự đoán
        nan_val = t("NaN (vắng)", "NaN (absent)")
        df_result.loc[df_result[col_gt].astype(str) == "nan", col_pred] = nan_val
        df_result.loc[df_result[col_gt].astype(str) == "nan", "Confidence"] = None

        n_correct = (df_result[col_check] == "✅").sum()
        n_total   = len(df_result)
        acc       = n_correct / n_total

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{t("Tổng sinh viên", "Total Students")}</div>
                <div class="value">{n_total}</div>
                <div class="sub">{t("phiếu điểm", "grading sheets")}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{t("Dự đoán đúng", "Correct Preds")}</div>
                <div class="value" style="color:#2e7d32">{n_correct}</div>
                <div class="sub">{t("ô điểm", "score cells")}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{t("Dự đoán sai", "Wrong Preds")}</div>
                <div class="value" style="color:#c62828">{n_total - n_correct}</div>
                <div class="sub">{t("ô điểm", "score cells")}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{t("Accuracy", "Accuracy")}</div>
                <div class="value">{acc:.1%}</div>
                <div class="sub">{t("trên phiếu thật", "on real sheets")}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Layout: bảng + ảnh
        col_table, col_img = st.columns([3, 2])

        with col_table:
            st.markdown(f'<div class="section-title">{t("Bảng kết quả", "Result Table")}</div>', unsafe_allow_html=True)

            # Highlight
            def highlight_row(row):
                base = [""] * len(row)
                idx_pred = df_result.columns.get_loc(col_pred)
                idx_gt   = df_result.columns.get_loc(col_gt)
                if row.iloc[idx_pred] == row.iloc[idx_gt]:
                    base[idx_pred] = "background-color: #e8f5e9; color: #2e7d32; font-weight:600"
                else:
                    base[idx_pred] = "background-color: #ffebee; color: #c62828; font-weight:600"
                return base

            df_show = df_result.drop(columns=[col_gt])
            styled  = df_show.style.apply(highlight_row, axis=1)
            st.dataframe(styled, height=480, use_container_width=True)

            csv_bytes = df_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                t("⬇️  Xuất kết quả CSV", "⬇️  Export CSV"),
                csv_bytes,
                "phieu_01_result.csv",
                "text/csv",
                use_container_width=True
            )

        with col_img:
            st.markdown(f'<div class="section-title">{t("Ảnh phiếu điểm", "Grading Sheet Image")}</div>', unsafe_allow_html=True)
            page_01 = BASE / "Test_Phieu_01" / "page_01.png"
            if page_01.exists():
                st.image(str(page_01), caption=t("Trang 1", "Page 1"), use_column_width=True)
            page_02 = BASE / "Test_Phieu_01" / "page_02.png"
            if page_02.exists():
                st.image(str(page_02), caption=t("Trang 2", "Page 2"), use_column_width=True)
    else:
        st.warning(t("Chưa tìm thấy dữ liệu Test_Phieu_01 trong thư mục.", "Test_Phieu_01 data not found in the directory."))
        st.write("DEBUG INFO: BASE=", str(BASE))
        st.write("Does csv_print_path exist?", csv_print_path.exists())
        st.write("Does csv_labels_path exist?", csv_labels_path.exists())
        st.write("Contents of BASE folder:")
        st.write([str(p.name) for p in BASE.iterdir()] if BASE.exists() else "BASE not found")
        test_phieu_01_path = BASE / "Test_Phieu_01"
        st.write("Contents of Test_Phieu_01 folder:")
        st.write([str(p.name) for p in test_phieu_01_path.iterdir()] if test_phieu_01_path.exists() else "Test_Phieu_01 not found")

# ══════════════════════════════════════════════════════════════════
# TAB 2 – DỰ ĐOÁN THỰC TẾ
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-title">{t("Upload phiếu điểm PDF → Tự động nhận diện Đ.số", "Upload PDF Grading Sheet → Auto Recognize Scores")}</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        📌 {t('App sử dụng tọa độ bounding box từ phiếu mẫu để tự động crop từng ô điểm.', 'App uses bounding box coordinates from the template to auto-crop cells.')}<br>
        {t('Phiếu upload cần cùng mẫu in với phiếu 01.', 'Uploaded sheet must have the same layout as template 01.')}
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(t("Chọn file PDF phiếu điểm", "Select PDF Grading Sheet"), type=["pdf"],
                                 label_visibility="collapsed")

    if uploaded:
        with st.spinner(t("Đang xử lý PDF...", "Processing PDF...")):
            pages = pdf_to_images(uploaded.read())

        st.success(t(f"✅ Đã load {len(pages)} trang", f"✅ Loaded {len(pages)} pages"))

        ref_w, ref_h = ref_size
        page_w, page_h = pages[0].size
        sx = page_w / ref_w
        sy = page_h / ref_h

        page_map = {
            "page_01": pages[0],
            "page_02": pages[1] if len(pages) > 1 else pages[0]
        }

        col_prev, col_res = st.columns([2, 3])
        with col_prev:
            st.markdown(f'<div class="section-title">{t("Preview trang 1", "Preview Page 1")}</div>',
                        unsafe_allow_html=True)
            st.image(pages[0], use_column_width=True)

        with col_res:
            st.markdown(f'<div class="section-title">{t("Kết quả dự đoán", "Prediction Results")}</div>',
                        unsafe_allow_html=True)

            with st.spinner(t("Đang nhận diện...", "Recognizing...")):
                results = []
                for page_key, page_boxes in boxes_tmpl.items():
                    page_img = page_map.get(page_key, pages[0])
                    for stt, box in page_boxes.items():
                        crop_img = crop_cell(page_img, box, sx, sy)
                        pred, conf, _ = predict(crop_img)
                        results.append({
                            t("STT", "No."): int(stt),
                            t("Đ.số dự đoán", "Pred Score"): pred,
                            "Confidence": round(conf, 3)
                        })

            df_pred = pd.DataFrame(results).sort_values(t("STT", "No.")).reset_index(drop=True)

            # Confidence color
            def conf_color(val):
                if val >= 0.9:   return "color: #2e7d32; font-weight:600"
                elif val >= 0.7: return "color: #f57c00; font-weight:600"
                else:            return "color: #c62828; font-weight:600"

            st.dataframe(
                df_pred.style.applymap(conf_color, subset=["Confidence"]),
                height=420,
                use_container_width=True
            )

            avg_conf = df_pred["Confidence"].mean()
            st.markdown(f"""
            <div class="metric-card" style="margin-top:12px">
                <div class="label">{t("Độ tin cậy TB", "Avg. Confidence")}</div>
                <div class="value">{avg_conf:.1%}</div>
            </div>""", unsafe_allow_html=True)

            csv2 = df_pred.to_csv(index=False).encode("utf-8-sig")
            st.download_button(t("⬇️  Xuất CSV", "⬇️  Export CSV"), csv2,
                               "phieu_pred.csv", "text/csv",
                               use_container_width=True)

        # Crop grid
        st.markdown(f'<div class="section-title">{t("Các ô điểm đã crop", "Cropped Score Cells")}</div>',
                    unsafe_allow_html=True)
        cols = st.columns(10)
        all_boxes_flat = []
        for pk, pb in boxes_tmpl.items():
            for stt, box in pb.items():
                all_boxes_flat.append((int(stt), pk, box))
        all_boxes_flat.sort(key=lambda x: x[0])

        for i, (stt, pk, box) in enumerate(all_boxes_flat):
            crop_img = crop_cell(page_map.get(pk, pages[0]), box, sx, sy)
            pred_row = df_pred[df_pred[t("STT", "No.")] == stt]
            pred_val = pred_row[t("Đ.số dự đoán", "Pred Score")].values[0] if len(pred_row) else "?"
            conf_val = pred_row["Confidence"].values[0] if len(pred_row) else 0
            cols[i % 10].image(crop_img,
                               caption=f"{stt}: {pred_val}\n({conf_val:.0%})",
                               use_column_width=True)

    else:
        st.markdown(f"""
        <div style="text-align:center; padding:60px; color:#adb5bd; 
                    border: 2px dashed #e0e4ef; border-radius:16px; margin-top:20px">
            <div style="font-size:3rem; margin-bottom:12px">📄</div>
            <div style="font-size:1rem; font-weight:500">{t('Kéo thả hoặc chọn file PDF phiếu điểm', 'Drag and drop or select a PDF grading sheet')}</div>
            <div style="font-size:0.82rem; margin-top:6px">{t('Hỗ trợ phiếu điểm mẫu E0330111', 'Supports grading sheet template E0330111')}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 – DỰ ĐOÁN ĐƠN LẺ
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-title">{t("Upload ảnh ô điểm bất kỳ để dự đoán", "Upload any score cell image for prediction")}</div>',
                unsafe_allow_html=True)

    img_file = st.file_uploader(t("Chọn ảnh ô điểm (PNG/JPG)", "Select score cell image (PNG/JPG)"),
                                 type=["png", "jpg", "jpeg"],
                                 label_visibility="collapsed")

    if img_file:
        img = Image.open(img_file).convert("RGB")
        pred, conf, top3 = predict(img)

        with st.spinner(t("Đang tính GradCAM...", "Calculating GradCAM...")):
            overlay_arr, _ = compute_gradcam(img)

        col_orig, col_cam, col_result = st.columns([1, 1, 1.2])

        with col_orig:
            st.markdown(f'<div class="section-title">{t("Ảnh gốc", "Original Image")}</div>',
                        unsafe_allow_html=True)
            st.image(img, use_column_width=True)

        with col_cam:
            st.markdown(f'<div class="section-title">{t("GradCAM – Vùng model chú ý", "GradCAM – Model Attention")}</div>',
                        unsafe_allow_html=True)
            st.image(Image.fromarray(overlay_arr), use_column_width=True)
            st.caption(t("Màu đỏ = vùng ảnh hưởng nhiều nhất đến dự đoán", "Red = Most influential region for prediction"))

        with col_result:
            st.markdown(f'<div class="section-title">{t("Kết quả dự đoán", "Prediction Results")}</div>',
                        unsafe_allow_html=True)

            conf_color_class = "#2e7d32" if conf >= 0.9 else ("#f57c00" if conf >= 0.7 else "#c62828")
            st.markdown(f"""
            <div class="pred-box">
                <div class="score">{pred}</div>
                <div class="conf">Confidence: 
                    <span style="color:{conf_color_class}">{conf:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**{t('Top-3 dự đoán:', 'Top-3 predictions:')}**")
            for label, prob in top3:
                bar_w = int(prob * 100)
                is_top = label == pred
                bar_color = "#1a1a2e" if is_top else "#c8d0e0"
                st.markdown(f"""
                <div class="bar-row">
                    <div class="bar-label" style="{'color:#1a1a2e;font-weight:700' if is_top else 'color:#8892a4'}">{label}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{bar_w}%; background:{bar_color}"></div>
                    </div>
                    <div class="bar-pct">{prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            # Device info
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.78rem; color:#adb5bd; text-align:center">
                Model: EfficientNet-B0 &nbsp;·&nbsp;
                Device: {'CUDA ⚡' if DEVICE.type == 'cuda' else 'CPU'}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="text-align:center; padding:80px; color:#adb5bd;
                    border: 2px dashed #e0e4ef; border-radius:16px; margin-top:20px">
            <div style="font-size:3rem; margin-bottom:12px">🖼️</div>
            <div style="font-size:1rem; font-weight:500">{t('Upload ảnh ô điểm viết tay', 'Upload handwritten score cell image')}</div>
            <div style="font-size:0.82rem; margin-top:6px">{t('PNG, JPG · Kích thước bất kỳ', 'PNG, JPG · Any size')}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    ScoreVision &nbsp;·&nbsp; {t('Đồ án chuyên ngành', 'Specialized Project')} &nbsp;·&nbsp;
    EfficientNet-B0 · 41 classes · 205,000 synthetic samples
</div>
""", unsafe_allow_html=True)