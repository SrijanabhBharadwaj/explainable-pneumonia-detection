"""Reproducible chest X-ray pneumonia experiments and explainability helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def download_dataset() -> Path:
    """Download the public Kaggle dataset with KaggleHub and return its root."""
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Install kagglehub or download the dataset manually; see README.md."
        ) from error
    root = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
    return find_dataset_root(root)


def find_dataset_root(path: str | Path) -> Path:
    """Locate the folder containing the original train/val/test directories."""
    path = Path(path).expanduser().resolve()
    candidates = [path, *path.rglob("*")]
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / split).is_dir() for split in SPLITS):
            return candidate
    raise FileNotFoundError(
        f"Could not find train/val/test folders under {path}. "
        "Expected the original Kermany directory structure."
    )


def image_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(
        item for item in folder.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_dataset_layout(root: str | Path) -> Path:
    root = find_dataset_root(root)
    for split in SPLITS:
        observed = {item.name.upper() for item in (root / split).iterdir() if item.is_dir()}
        if set(CLASS_NAMES) - observed:
            raise ValueError(
                f"{root / split} must contain NORMAL and PNEUMONIA directories; "
                f"found {sorted(observed)}"
            )
    return root


def inspect_dataset(root: str | Path, make_plots: bool = True) -> dict[str, Any]:
    """Count files and inspect dimensions, readability, and exact cross-split duplicates."""
    root = validate_dataset_layout(root)
    rows: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []
    unreadable: list[str] = []
    exact_hashes: dict[str, list[tuple[str, str]]] = {}

    for split in SPLITS:
        for class_name in CLASS_NAMES:
            files = image_files(root / split / class_name)
            rows.append({"split": split, "class": class_name, "images": len(files)})
            for file_path in files:
                image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    unreadable.append(str(file_path))
                    continue
                height, width = image.shape[:2]
                dimensions.append({"split": split, "class": class_name,
                                   "width": width, "height": height})
                digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
                exact_hashes.setdefault(digest, []).append((split, str(file_path)))

    counts = pd.DataFrame(rows)
    dimension_table = pd.DataFrame(dimensions)
    duplicates = [
        {"sha256": digest, "files": locations}
        for digest, locations in exact_hashes.items()
        if len({split for split, _ in locations}) > 1
    ]
    total = int(counts["images"].sum())
    class_totals = counts.groupby("class")["images"].sum().reindex(CLASS_NAMES, fill_value=0)
    minority = int(class_totals.min())
    imbalance_ratio = (float(class_totals.max()) / minority) if minority else float("inf")
    summary = {
        "root": str(root),
        "counts": counts,
        "total_images": total,
        "class_totals": class_totals,
        "imbalance_ratio_majority_to_minority": imbalance_ratio,
        "dimensions": dimension_table,
        "unreadable_files": unreadable,
        "exact_cross_split_duplicates": duplicates,
    }

    print(f"Dataset root: {root}")
    print(f"Total images discovered: {total:,}")
    print("Counts by original split and class:")
    print(counts.pivot(index="split", columns="class", values="images").fillna(0).astype(int))
    print("\nClass totals:")
    print(class_totals.to_string())
    print(f"Majority/minority ratio: {imbalance_ratio:.3f}:1")
    if dimension_table.empty:
        print("No readable image dimensions found.")
    else:
        print("\nReadable image dimension summary (width x height):")
        print(dimension_table[["width", "height"]].describe().round(1))
    print(f"Unreadable image files: {len(unreadable)}")
    print(f"Exact byte-identical duplicates spanning splits: {len(duplicates)}")
    print(
        "Note: exact hashes cannot rule out patient-level leakage. The released files "
        "do not provide reliable patient identifiers for a subject-wise audit."
    )

    if make_plots:
        totals_by_split = counts.pivot(index="split", columns="class", values="images").fillna(0)
        totals_by_split.plot(kind="bar", figsize=(8, 4), color=["#4c78a8", "#e45756"])
        plt.title("Images in supplied dataset splits")
        plt.ylabel("Number of files")
        plt.xlabel("Original split")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
        if not dimension_table.empty:
            plt.figure(figsize=(7, 4))
            plt.scatter(dimension_table["width"], dimension_table["height"], alpha=0.25, s=12)
            plt.title("Image dimensions before resizing")
            plt.xlabel("Width (pixels)")
            plt.ylabel("Height (pixels)")
            plt.tight_layout()
            plt.show()
    return summary


def show_samples(root: str | Path, per_class: int = 4, split: str = "train") -> None:
    root = validate_dataset_layout(root)
    figure, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(3 * per_class, 6))
    for row, class_name in enumerate(CLASS_NAMES):
        paths = image_files(root / split / class_name)[:per_class]
        for column in range(per_class):
            axis = axes[row, column] if per_class > 1 else axes[row]
            axis.axis("off")
            if column < len(paths):
                image = cv2.cvtColor(cv2.imread(str(paths[column])), cv2.COLOR_BGR2RGB)
                axis.imshow(image, cmap="gray")
                axis.set_title(class_name)
    plt.tight_layout()
    plt.show()


def get_class_weights(root: str | Path) -> dict[int, float]:
    root = validate_dataset_layout(root)
    labels: list[int] = []
    for label, class_name in enumerate(CLASS_NAMES):
        labels.extend([label] * len(image_files(root / "train" / class_name)))
    weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=np.asarray(labels)
    )
    return {int(index): float(weight) for index, weight in enumerate(weights)}


def build_datasets(
    root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    augment: bool = True,
    horizontal_flip: bool = False,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load the supplied splits; only train is shuffled and augmented."""
    root = validate_dataset_layout(root)
    options = dict(
        labels="inferred", label_mode="binary", color_mode="rgb",
        image_size=(image_size, image_size), batch_size=batch_size,
        pad_to_aspect_ratio=True,
    )
    train = tf.keras.utils.image_dataset_from_directory(
        root / "train", shuffle=True, seed=seed, **options
    )
    class_names = tuple(name.upper() for name in train.class_names)
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected class order {CLASS_NAMES}, received {class_names}")
    validation = tf.keras.utils.image_dataset_from_directory(
        root / "val", shuffle=False, **options
    )
    test = tf.keras.utils.image_dataset_from_directory(
        root / "test", shuffle=False, **options
    )
    if tuple(name.upper() for name in validation.class_names) != CLASS_NAMES:
        raise ValueError("Validation class directories do not match training classes")
    if tuple(name.upper() for name in test.class_names) != CLASS_NAMES:
        raise ValueError("Test class directories do not match training classes")

    if augment:
        augmentation_layers = [
            tf.keras.layers.RandomRotation(0.025, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomZoom(0.08, fill_mode="reflect", seed=seed),
        ]
        if horizontal_flip:
            augmentation_layers.insert(0, tf.keras.layers.RandomFlip("horizontal", seed=seed))
        augmenter = tf.keras.Sequential(augmentation_layers)
        train = train.map(
            lambda images, labels: (augmenter(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return tuple(dataset.prefetch(tf.data.AUTOTUNE) for dataset in (train, validation, test))


def build_baseline(image_size: int = 224) -> tf.keras.Model:
    inputs = tf.keras.Input((image_size, image_size, 3), name="xray")
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(name="last_activation")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(
    architecture: str, image_size: int = 224, dropout: float = 0.3
) -> tuple[tf.keras.Model, tf.keras.Model]:
    name = architecture.lower().replace("-", "")
    constructors = {
        "densenet121": tf.keras.applications.DenseNet121,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
    }
    if name not in constructors:
        raise ValueError("architecture must be DenseNet121 or EfficientNetB0")
    backbone = constructors[name](
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input((image_size, image_size, 3), name="xray_rgb_0_255")
    if name == "densenet121":
        x = tf.keras.applications.densenet.preprocess_input(inputs)
    else:
        x = inputs
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    return model, backbone


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
                 tf.keras.metrics.AUC(name="pr_auc", curve="PR")],
    )


def callbacks_for(output_dir: str | Path, run_name: str, patience: int = 5) -> list[tf.keras.callbacks.Callback]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / f"{run_name}.keras", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=max(2, patience // 2), min_lr=1e-7
        ),
    ]


def train_experiment(
    model: tf.keras.Model,
    train: tf.data.Dataset,
    validation: tf.data.Dataset,
    output_dir: str | Path,
    run_name: str,
    epochs: int = 20,
    class_weight: dict[int, float] | None = None,
    learning_rate: float = 1e-3,
    patience: int = 5,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    compile_model(model, learning_rate)
    history = model.fit(
        train, validation_data=validation, epochs=epochs,
        class_weight=class_weight, callbacks=callbacks_for(output_dir, run_name, patience),
    )
    best_model = tf.keras.models.load_model(Path(output_dir) / f"{run_name}.keras")
    return best_model, history


def fine_tune(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    train: tf.data.Dataset,
    validation: tf.data.Dataset,
    output_dir: str | Path,
    run_name: str,
    epochs: int = 10,
    unfreeze_last: int = 30,
    class_weight: dict[int, float] | None = None,
    learning_rate: float = 1e-5,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - unfreeze_last)
    for layer in backbone.layers[:cutoff]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, learning_rate)
    history = model.fit(
        train, validation_data=validation, epochs=epochs, class_weight=class_weight,
        callbacks=callbacks_for(output_dir, f"{run_name}_finetuned"),
    )
    best_model = tf.keras.models.load_model(Path(output_dir) / f"{run_name}_finetuned.keras")
    return best_model, history


def plot_history(histories: list[tuple[str, tf.keras.callbacks.History]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories:
        for axis, metric, title in ((axes[0], "loss", "Loss"),
                                    (axes[1], "accuracy", "Accuracy")):
            axis.plot(history.history.get(metric, []), label=f"{name} train")
            axis.plot(history.history.get(f"val_{metric}", []), linestyle="--",
                      label=f"{name} validation")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.legend()
    plt.tight_layout()
    plt.show()


def labels_and_predictions(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate([np.asarray(y).reshape(-1) for _, y in dataset]).astype(int)
    probabilities = model.predict(dataset, verbose=0).reshape(-1)
    return labels, probabilities


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray,
                           threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else float("nan"),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def evaluate_models(
    models: dict[str, tf.keras.Model], dataset: tf.data.Dataset,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        y_true, probabilities = labels_and_predictions(model, dataset)
        metrics = classification_metrics(y_true, probabilities)
        results[name] = {"y_true": y_true, "probabilities": probabilities, "metrics": metrics}
        print(f"\n{name}")
        print(pd.Series(metrics).to_string())
        matrix = confusion_matrix(y_true, probabilities >= 0.5, labels=[0, 1])
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
        plt.title(f"{name} confusion matrix")
        plt.tight_layout()
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(output_dir) / f"{name}_confusion_matrix.png", dpi=160)
        plt.show()
    table = pd.DataFrame({name: item["metrics"] for name, item in results.items()}).T
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        table.to_csv(Path(output_dir) / "model_comparison.csv", index_label="model")
    return table, results


def test_file_paths(root: str | Path) -> list[Path]:
    root = validate_dataset_layout(root)
    return [path for class_name in CLASS_NAMES
            for path in image_files(root / "test" / class_name)]


def analyze_errors(
    model: tf.keras.Model, root: str | Path, dataset: tf.data.Dataset,
    per_category: int = 4,
) -> pd.DataFrame:
    y_true, probabilities = labels_and_predictions(model, dataset)
    paths = test_file_paths(root)
    if len(paths) != len(y_true):
        raise ValueError("Test file order/count does not match the unshuffled TensorFlow dataset")
    predictions = (probabilities >= 0.5).astype(int)
    categories = np.where(y_true == 1, np.where(predictions == 1, "TP", "FN"),
                          np.where(predictions == 1, "FP", "TN"))
    table = pd.DataFrame({"path": [str(path) for path in paths], "true_label": y_true,
                          "predicted_label": predictions, "probability_pneumonia": probabilities,
                          "category": categories})
    figure, axes = plt.subplots(4, per_category, figsize=(3 * per_category, 12))
    for row, category in enumerate(("TP", "TN", "FP", "FN")):
        subset = table[table["category"] == category].head(per_category)
        for column in range(per_category):
            axis = axes[row, column] if per_category > 1 else axes[row]
            axis.axis("off")
            if column < len(subset):
                example = subset.iloc[column]
                image = cv2.cvtColor(cv2.imread(example["path"]), cv2.COLOR_BGR2RGB)
                axis.imshow(image)
                axis.set_title(
                    f"{category} | p={example['probability_pneumonia']:.3f}\n"
                    f"True: {CLASS_NAMES[int(example['true_label'])]}"
                )
    plt.tight_layout()
    plt.show()
    print("False negatives are pneumonia cases predicted as NORMAL; review these first.")
    return table


def _load_rgb(path: str | Path, image_size: int = 224) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return cv2.cvtColor(cv2.resize(image, (image_size, image_size)), cv2.COLOR_BGR2RGB)


def gradcam_heatmap(
    model: tf.keras.Model, image_rgb: np.ndarray, backbone: tf.keras.Model | None = None,
    layer_name: str | None = None,
) -> np.ndarray:
    input_tensor = tf.convert_to_tensor(image_rgb[None].astype(np.float32))
    with tf.GradientTape() as tape:
        if model.name == "baseline_cnn":
            target_layer = model.get_layer(layer_name or "last_activation")
            gradient_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])
            feature_maps, predictions = gradient_model(input_tensor, training=False)
        else:
            backbone_name = backbone.name if backbone is not None else model.name
            backbone = model.get_layer(backbone_name)
            if layer_name is None:
                layer_name = "top_conv" if "efficientnet" in model.name else "conv5_block16_2_conv"
            target_layer = backbone.get_layer(layer_name)
            feature_model = tf.keras.Model(backbone.input, [target_layer.output, backbone.output])
            if "densenet" in model.name:
                input_tensor = tf.keras.applications.densenet.preprocess_input(input_tensor)
            feature_maps, backbone_output = feature_model(input_tensor, training=False)
            head = model.get_layer("global_pool")(backbone_output)
            head = model.get_layer("head_dropout")(head, training=False)
            predictions = model.get_layer("pneumonia_probability")(head)
        score = predictions[:, 0]
    gradients = tape.gradient(score, feature_maps)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = tf.image.resize(heatmap[..., None], image_rgb.shape[:2]).numpy().squeeze()
    return heatmap


def show_gradcam_cases(
    model: tf.keras.Model, errors: pd.DataFrame, backbone: tf.keras.Model | None = None,
    categories: tuple[str, ...] = ("TP", "TN", "FP", "FN"), image_size: int = 224,
) -> None:
    figure, axes = plt.subplots(len(categories), 3, figsize=(12, 4 * len(categories)))
    for row, category in enumerate(categories):
        subset = errors[errors["category"] == category]
        if subset.empty:
            for axis in axes[row]:
                axis.axis("off")
            axes[row, 0].text(0.05, 0.5, f"No {category} example at threshold 0.5")
            continue
        example = subset.iloc[0]
        image = _load_rgb(example["path"], image_size)
        heatmap = gradcam_heatmap(model, image, backbone)
        color_map = plt.get_cmap("jet")(heatmap)[..., :3]
        overlay = np.clip(0.55 * (image / 255.0) + 0.45 * color_map, 0, 1)
        prediction = CLASS_NAMES[int(example["predicted_label"])]
        title = f"{category}: predicted {prediction}, P(pneumonia)={example['probability_pneumonia']:.3f}"
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Original\n" + title)
        axes[row, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title("Grad-CAM")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        for axis in axes[row]:
            axis.axis("off")
    plt.tight_layout()
    plt.show()


def _robustness_variant(image: np.ndarray, variation: str) -> np.ndarray:
    if variation == "brightness_down":
        return np.clip(image * 0.8, 0, 255).astype(np.uint8)
    if variation == "brightness_up":
        return np.clip(image * 1.2, 0, 255).astype(np.uint8)
    if variation == "contrast_down":
        return np.clip((image - 127.5) * 0.8 + 127.5, 0, 255).astype(np.uint8)
    if variation == "contrast_up":
        return np.clip((image - 127.5) * 1.2 + 127.5, 0, 255).astype(np.uint8)
    if variation == "gaussian_noise":
        seed = int.from_bytes(hashlib.sha256(image.tobytes()).digest()[:4], "big")
        noise = np.random.default_rng(seed).normal(0, 8, image.shape)
        return np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    if variation == "slight_blur":
        return cv2.GaussianBlur(image, (3, 3), 0.5)
    if variation.startswith("resolution_"):
        resolution = int(variation.split("_")[1])
        low_resolution = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
        return cv2.resize(low_resolution, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"Unknown variation: {variation}")


def robustness_evaluation(
    model: tf.keras.Model, root: str | Path, image_size: int = 224,
    batch_size: int = 32,
) -> pd.DataFrame:
    paths = test_file_paths(root)
    labels = np.array([0 if "NORMAL" in path.parts else 1 for path in paths], dtype=int)
    variations = (
        "original", "brightness_down", "brightness_up", "contrast_down", "contrast_up",
        "gaussian_noise", "slight_blur", "resolution_128", "resolution_320",
    )
    rows = []
    for variation in variations:
        images = []
        for path in paths:
            image = _load_rgb(path, image_size)
            if variation != "original":
                image = _robustness_variant(image, variation)
            images.append(image)
        probabilities = model.predict(np.asarray(images, dtype=np.float32),
                                      batch_size=batch_size, verbose=0).reshape(-1)
        metrics = classification_metrics(labels, probabilities)
        rows.append({"variation": variation, **metrics})
    return pd.DataFrame(rows)


def save_run_metadata(
    output_dir: str | Path, root: str | Path, image_size: int, batch_size: int,
    seed: int, class_weights: dict[int, float],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_root": str(Path(root).resolve()), "image_size": image_size,
        "batch_size": batch_size, "seed": seed,
        "class_names_label_order": {"0": "NORMAL", "1": "PNEUMONIA"},
        "class_weights": class_weights,
        "tensorflow_version": tf.__version__,
        "note": "Metrics must be generated by running the experiments; no results are prefilled.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")