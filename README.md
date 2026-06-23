# ScoreVision - Automated Score Recognition System

🎓 **ScoreVision** is an end-to-end Machine Learning pipeline and Web Application designed to automatically recognize, extract, and grade handwritten scores from physical examination papers and PDF documents.

## 🌟 Key Features
- **End-to-End Pipeline**: From synthetic data generation to model training, evaluation, and deployment.
- **Synthetic Data Generation**: Generated a custom dataset of >200,000 synthetic handwritten score samples to overcome the lack of real-world annotated data.
- **Model Training & Benchmarking**: Trained and benchmarked multiple CNN and Vision Transformer architectures including **EfficientNet-B0**, **MobileNetV3**, and **DeiT-Tiny**.
- **Explainable AI (XAI)**: Integrated **Grad-CAM** to visualize and explain model predictions, ensuring transparency in grading.
- **Interactive Web App**: A beautiful, responsive web interface built with **Streamlit** that allows users to upload images or PDFs and get instant score predictions with confidence metrics.

## 📁 Project Structure
The project is organized into sequential Jupyter Notebooks documenting the entire research and development process, alongside the deployment source code:

- `00_project_audit.ipynb` - Initial data auditing and exploration.
- `01_synthetic_generation.ipynb` - Logic for generating synthetic handwritten score datasets.
- `02_dataset_build.ipynb` - Dataset splitting (Train/Val/Test) and formatting.
- `03_train_cnn.ipynb` - Model training and hyperparameter tuning.
- `04_evaluation.ipynb` - Model evaluation on test sets.
- `05_xai_gradcam.ipynb` - Explainable AI analysis using Grad-CAM.
- `06_report_supplement_checks.ipynb` - Supplementary checks and validation.
- `07_report_extra_metrics.ipynb` - Deep dive into evaluation metrics and performance comparisons.
- `streamlit_app.py` - The main source code for the interactive Streamlit Web Application.

## 🚀 How to Run the Web App Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ScoreVision.git
   cd ScoreVision
   ```

2. **Install dependencies**
   Make sure you have Python 3.8+ installed. Run the following command to install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run streamlit_app.py
   ```
   The app will open automatically in your browser at `http://localhost:8501`.

## 📊 Results Summary
*   **Best Model**: EfficientNet-B0 and DeiT-Tiny showed excellent performance.
*   **Accuracy**: Achieved highly competitive accuracy on real-world test sets.
*   **Speed**: Inference time is fully optimized for real-time applications (approx. 2-6ms per image).

## 🏆 Kaggle Resources
- **[Synthetic Handwritten Score Dataset]((Link-to-your-kaggle-dataset))**: Access the 200k+ generated dataset used to train these models.
- **[Training Notebooks]((Link-to-your-kaggle-notebook))**: View the full training process, EDA, and Grad-CAM visualizations interactively on Kaggle.

---
*Developed as part of a specialized university project. Ready for Data Science and Machine Learning engineering roles.*
