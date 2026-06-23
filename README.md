# ScoreVision - Handwritten Score Recognition System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nhandienchusoviettaytuhinhanhduytung.streamlit.app)

**ScoreVision** is an end-to-end Machine Learning pipeline and Web Application designed to automatically recognize, extract, and digitize handwritten scores from physical examination grade sheets. 

*Developed as a specialized university project by Nguyễn Đinh Tùng and Nguyễn Phạm Duy, under the supervision of TS. Phùng Thế Bảo.*

---

## 🎯 Project Objectives & Scope
The system aims to digitize the manual grading entry process in educational environments, reducing human error and saving time. 
- **Scope**: Recognizes handwritten scores within pre-defined templates (template-based cropping). Supports 41 valid score classes (0.0 to 10.0 with typical intervals).
- **Not a General OCR**: Designed specifically for grade sheets matching the configured template.

### 🌟 Key Features
- **Template-based Cropping**: Automatically extracts individual score cells from full-page PDFs.
- **NaN Detection**: Intelligently identifies absent, crossed-out, or empty cells and returns `NaN`.
- **Confidence Metrics**: Flags low-confidence predictions for manual human review.
- **Explainable AI (XAI)**: Integrates **Grad-CAM** to visualize which parts of the image influenced the model's decision.
- **CSV Export**: Instantly exports digitized scores to CSV for database integration.

---

## 📊 Data Pipeline & Synthetic Generation
Due to the lack of annotated real-world grade sheets, we built a robust synthetic data generation pipeline:
- **Raw Source**: [Handwritten Digits 0-9 Kaggle Dataset](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9) (21,600 images).
- **Synthetic Expansion**: Generated **>200,000 synthetic handwritten score images** spanning 41 decimal score classes.
- **Augmentations**: Simulated real-world noise, blur, table lines, tight crops, and varying decimal point styles (dots/commas) to minimize the domain gap.

---

## 🧠 Model Training & Deployment
We trained and benchmarked 6 deep learning architectures to evaluate performance on both synthetic and real-world external test data:
1. CNN Baseline
2. ResNet18
3. MobileNetV3-Small
4. EfficientNet-B0
5. EfficientNet-B3
6. DeiT-Tiny

**Deployment Selection: `EfficientNet-B0`**
While `DeiT-Tiny` achieved the highest absolute accuracy during external testing, `EfficientNet-B0` was selected for the final Streamlit deployment due to its optimal balance between accuracy, minimal model size, and fast real-time inference speed suitable for web environments.

---

## 💻 Streamlit Web Application
The deployed application features 3 main interactive modules:
1. **Full Demo (Phieu 01)**: Evaluates a predefined real-world test sheet, comparing predictions against ground truth, handling NaN cells, and exporting to CSV.
2. **Real PDF Prediction**: Upload a full PDF grade sheet. The app renders pages, applies template-based cropping, predicts scores using EfficientNet-B0, and outputs a downloadable CSV.
3. **Single Crop Prediction**: Upload a single score image crop to receive Top-3 predictions, confidence scores, and a Grad-CAM visualization.

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tungmessi/ScoreVision.git
   cd ScoreVision
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run streamlit_app.py
   ```
   *The app will automatically open in your browser at `http://localhost:8501`.*

---

## 📂 Project Structure
The repository contains the source code for the Streamlit app and 6 sequential Jupyter Notebooks documenting the entire ML pipeline:
- `00_project_audit.ipynb`: Preparation and formatting of external real-world test set labels.
- `01_synthetic_generation.ipynb`: Logic for generating the synthetic handwritten score dataset.
- `02_dataset_build.ipynb`: Dataset splitting (Train/Val/Test) and formatting.
- `03_train_cnn.ipynb`: Training and hyperparameter tuning for the 6 deep learning models.
- `04_evaluation.ipynb`: Comprehensive model evaluation on synthetic and real test sets.
- `05_xai_gradcam.ipynb`: Explainable AI analysis using Grad-CAM.
- `streamlit_app.py`: Main application source code.

---

## 🔮 Future Work
- Integration of an automated table and cell detection module (e.g., YOLO) to remove the reliance on rigid templates.
- OCR integration for machine-printed text on the grade sheets (student ID, name).
- Expansion of the real-world dataset to further minimize the domain gap.
- Direct database integration for seamless score management.
