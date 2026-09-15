import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
N_PCA_COMPONENTS = 125
PCA_BATCH_SIZE = 2000          # images per IncrementalPCA batch -- keeps memory use low
RANDOM_STATE = 42

CLASSES = ["disc", "spiral", "elliptical", "round", "irregular"]

BASE_DIR = r"D:\galaxymorpho"
LABELED_CSV = f"{BASE_DIR}/gz2_labeled.csv"
FEATURE_CACHE = f"{BASE_DIR}/pca_features_cache_full.npz"


def _load_one_image(path_str):
    img = Image.open(path_str).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32).flatten() / 255.0


def _image_batches(df: pd.DataFrame, batch_size: int):
    """Yields (X_batch, y_batch) arrays, skipping any missing image files."""
    paths = df["image_path"].tolist()
    labels = df["label"].tolist()
    n = len(paths)
    n_batches = (n + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        X_list, y_list = [], []
        for i in range(start, end):
            p = Path(paths[i])
            if not p.exists():
                continue
            try:
                X_list.append(_load_one_image(p))
                y_list.append(labels[i])
            except Exception as e:
                print(f"Skipping {p}: {e}")
        print(f"    batch {b + 1}/{n_batches} ({len(X_list)} images loaded)", flush=True)
        if X_list:
            yield np.array(X_list, dtype=np.float32), np.array(y_list)


def load_images_and_labels(df: pd.DataFrame):
    X_parts, y_parts = [], []
    for Xb, yb in _image_batches(df, batch_size=2000):
        X_parts.append(Xb)
        y_parts.append(yb)
    if not X_parts:
        raise RuntimeError(
            "No images were found on disk for any of the rows given. "
            "This almost always means image_path values are stale (pointing at an "
            "old/incomplete images folder) or IMAGES_DIR in 00_data_prep.py is wrong. "
            "Fix IMAGES_DIR, then re-run 00_data_prep.py to regenerate gz2_labeled.csv "
            "with correct paths -- re-extracting the zip alone does NOT update the CSV."
        )
    return np.vstack(X_parts), np.concatenate(y_parts)


def get_train_test_pca_features(use_cache: bool = True):
    
    if use_cache and Path(FEATURE_CACHE).exists():
        print(f"Loading cached PCA features from {FEATURE_CACHE}")
        data = np.load(FEATURE_CACHE, allow_pickle=True)
        print(f"  Train: {data['X_train'].shape[0]:,}  Test: {data['X_test'].shape[0]:,}")
        return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    df = pd.read_csv(LABELED_CSV)
    print(f"Full dataset: {len(df):,} rows")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
    )
    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

    ipca = IncrementalPCA(n_components=N_PCA_COMPONENTS, batch_size=PCA_BATCH_SIZE)

    print(f"Pass 1/2: fitting IncrementalPCA on {len(train_df):,} training images...")
    for Xb, yb in _image_batches(train_df, PCA_BATCH_SIZE):
        if Xb.shape[0] < N_PCA_COMPONENTS:
            continue  # IncrementalPCA needs each partial_fit batch >= n_components
        ipca.partial_fit(Xb)

    print(f"Pass 2/2: transforming {len(train_df):,} training images...")
    X_train_parts, y_train_parts = [], []
    for Xb, yb in _image_batches(train_df, PCA_BATCH_SIZE):
        X_train_parts.append(ipca.transform(Xb))
        y_train_parts.append(yb)
    X_train_p = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    print(f"Transforming {len(test_df):,} test images...")
    X_test_parts, y_test_parts = [], []
    for Xb, yb in _image_batches(test_df, PCA_BATCH_SIZE):
        X_test_parts.append(ipca.transform(Xb))
        y_test_parts.append(yb)
    X_test_p = np.vstack(X_test_parts)
    y_test = np.concatenate(y_test_parts)

    print(f"PCA: {N_PCA_COMPONENTS} components explain "
          f"{ipca.explained_variance_ratio_.sum():.3f} of variance")

    if use_cache:
        np.savez(FEATURE_CACHE, X_train=X_train_p, X_test=X_test_p,
                  y_train=y_train, y_test=y_test)
        print(f"Cached features to {FEATURE_CACHE} -- later scripts will load instantly.")

    return X_train_p, X_test_p, y_train, y_test


def get_full_pca_features():
    
    X_train_p, X_test_p, y_train, y_test = get_train_test_pca_features()
    X_p = np.vstack([X_train_p, X_test_p])
    y = np.concatenate([y_train, y_test])
    return X_p, y


def report_clustering(method_name: str, X_p, y, labels_by_k: dict):
   
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

    rows = []
    for k, labels in labels_by_k.items():
        sil = silhouette_score(X_p, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)
        print(f"k={k}: silhouette={sil:.3f}  ARI={ari:.3f}  NMI={nmi:.3f}")
        rows.append({"method": method_name, "k": k, "silhouette": sil, "ari": ari, "nmi": nmi})

    out = pd.DataFrame(rows)
    out_path = f"clustering_{method_name.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    return out


def save_confusion_matrix(model_name: str, y_test, preds, classes=None):
    
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if classes is None:
        classes = sorted(set(y_test) | set(preds))

    cm = confusion_matrix(y_test, preds, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "true"
    cm_df.columns.name = "predicted"

    csv_path = f"confusion_matrix_{model_name.lower()}.csv"
    cm_df.to_csv(csv_path)
    print(f"Saved confusion matrix to {csv_path}")
    print(cm_df)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} — Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    png_path = f"confusion_matrix_{model_name.lower()}.png"
    plt.savefig(png_path)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {png_path}")

    return cm_df


def report_and_save(model_name: str, y_test, preds):
    from sklearn.metrics import classification_report, accuracy_score
    import json

    acc = accuracy_score(y_test, preds)
    print(f"\n=== {model_name} (test accuracy = {acc:.4f}) ===")
    print(classification_report(y_test, preds))

    report = classification_report(y_test, preds, output_dict=True)
    result = {"model": model_name, "test_accuracy": acc, "report": report}

    out_path = f"results_{model_name.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_path}")
