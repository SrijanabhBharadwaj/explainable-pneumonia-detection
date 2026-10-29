# Explainable Pneumonia Detection from Chest X-Rays

An educational research pipeline for binary NORMAL/PNEUMONIA classification using the public Chest X-Ray Images (Pneumonia) dataset associated with Kermany et al. It includes dataset inspection, a CNN baseline, DenseNet121 and EfficientNet-B0 transfer learning, evaluation, error review, Grad-CAM, and controlled robustness tests.

**This is not a medical device.** Dataset labels and model predictions are not clinical diagnoses. Grad-CAM highlights gradient-based model sensitivity; it does not prove that a model localized disease or used clinically valid reasoning.

## Project files

- `streamlit_app.py`: Streamlit research demo for dataset status and X-ray inference.
- `pneumonia_pipeline.py`: reusable data, model, evaluation, explanation, and robustness functions.
- `requirements.txt`: pipeline and website dependencies.
- `outputs/`: local model checkpoints and generated results; excluded from Git.

## Run the website

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Place the trained `cpu_baseline_exploratory.keras` checkpoint in `outputs/`.
4. Start the website with `streamlit run streamlit_app.py`.

The checkpoint and dataset are not included in this repository. The website currently looks for the dataset at the local path configured in `streamlit_app.py`. Update that path for your machine. Checkpoints under `outputs/` are intentionally ignored by Git; configure separate model storage for deployment.

## Training in Google Colab

The narrated `pneumonia_transfer_gradcam.ipynb` workflow is not included in this source repository. If you have it, upload it alongside `pneumonia_pipeline.py` to Colab, or put both in a folder in Google Drive and update `PROJECT_DIR` in the notebook.

1. Choose **Runtime > Change runtime type > T4 GPU** (or another available GPU).
2. Run the dependency/setup and dataset cells. The notebook tries the public KaggleHub dataset `paultimothymooney/chest-xray-pneumonia` and locates the original `train`, `val`, and `test` folders. If Kaggle requires authentication, configure a Kaggle API token through Colab Secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`) or download the dataset from Kaggle and set `DATASET_ROOT` to the extracted directory.
3. Run the exploration cells before training. Counts, class ratios, unreadable files, dimensions, and duplicate checks are measured from the files actually present; no dataset totals or model scores are hard-coded.
4. Run the experiments in order. The test split is reserved for final comparisons, error analysis, and robustness reporting. Do not select thresholds, epochs, or models based on test outcomes.
5. Results and checkpoints are written under `outputs/`. Copy `outputs/` to persistent Drive storage if results must survive a Colab session reset.

Training all variants can take a while. The notebook exposes epoch, batch-size, and fine-tuning controls. Begin with fewer epochs to check the pipeline, then choose and record a compute budget before the final run. A fixed seed improves repeatability but does not guarantee bit-for-bit determinism across different GPUs, TensorFlow builds, and kernels.

## Dataset structure and safeguards

The expected archive has `train/`, `val/`, and `test/` folders, each containing `NORMAL/` and `PNEUMONIA/`. The project preserves those released splits and computes their size and label distribution at runtime. It does not move validation or test images into training. Augmentation is applied only to the training dataset.

Exact file hashes can identify byte-identical files crossing splits, but cannot establish patient independence, detect all re-encoded near-duplicates, or rule out source-level leakage. Filenames and image-level splits do not provide a dependable subject-level study design. Treat any split-level score as performance on this particular curated dataset, not external clinical validation.

The collection is useful for student work because it is public, compact enough for transfer-learning experiments, and has a simple two-class task. Its limitations include dataset and acquisition bias, uncertain patient-level independence, limited demographic and institution diversity, label/curation uncertainty, class imbalance, and lack of prospective external validation. Results cannot support clinical use.

## Preprocessing and experiment design

Images are decoded as RGB and resized to 224 x 224 while preserving aspect ratio by padding. Pixel input remains in the 0-255 range: DenseNet applies its ImageNet preprocessing, EfficientNet's built-in preprocessing handles its expected range, and the baseline rescales internally. Validation and test transformations are deterministic and contain no augmentation.

The augmentation defaults are deliberately mild rotation, translation, and zoom. Horizontal flipping is off by default because laterality and asymmetric devices can matter; it can be enabled as an explicitly recorded experiment. No vertical flip, strong crop, elastic deformation, or aggressive color transformation is used.

The notebook compares unweighted training, training with weights computed only from the training split, and augmentation. It also compares transfer-learning models using classification metrics beyond accuracy. Recall/sensitivity measures the proportion of pneumonia cases detected; false negatives are especially important to inspect, but a useful operating point must also account for false positives and clinical context. The default 0.5 threshold is a reporting convention, not a clinically validated decision threshold.

## Literature

- Kermany et al. (2018). “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.” *Cell*.
- Rajpurkar et al. (2017). “CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.”
- Selvaraju et al. (2017). “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.”
- Hasibuan et al. (2022). “A Multichannel EfficientNet Deep Learning-Based Stacking Ensemble Approach for Lung Disease Detection Using Chest X-Rays.”

The cited work motivates the task and methods; this project does not claim to reproduce their datasets, model configurations, or reported results.