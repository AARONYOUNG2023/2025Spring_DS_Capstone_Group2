import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def chexpert_label_text(report_text):
    """
    Applies the CheXpert labeler to a single string (report_text).
    Returns a dict of pathology->0/1.
    In practice, integrate the official CheXpert labeler or a local library.
    """
    import random
    pathologies = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
    labels = {p: random.randint(0, 1) for p in pathologies}
    return labels


def evaluate_clinical_accuracy(generated_texts, reference_texts):
    """
    Compare CheXpert pathology labels between generated reports and reference/ground truth.
    Returns dict with accuracy, F1, precision, recall for each pathology, plus macro averages.
    """

    pathologies = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

    y_true = []
    y_pred = []

    for gen_text, ref_text in zip(generated_texts, reference_texts):
        ref_labels = chexpert_label_text(ref_text)  # e.g., {"Cardiomegaly":1, "Edema":0, ...}
        gen_labels = chexpert_label_text(gen_text)

        y_true.append([ref_labels[p] for p in pathologies])
        y_pred.append([gen_labels[p] for p in pathologies])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    metrics = {}
    for i, pathology in enumerate(pathologies):
        pathology_y_true = y_true[:, i]
        pathology_y_pred = y_pred[:, i]

        p = precision_score(pathology_y_true, pathology_y_pred)
        r = recall_score(pathology_y_true, pathology_y_pred)
        f1 = f1_score(pathology_y_true, pathology_y_pred)
        acc = accuracy_score(pathology_y_true, pathology_y_pred)

        metrics[pathology] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "accuracy": acc
        }

    macro_precision = np.mean([metrics[p]["precision"] for p in pathologies])
    macro_recall = np.mean([metrics[p]["recall"] for p in pathologies])
    macro_f1 = np.mean([metrics[p]["f1"] for p in pathologies])
    macro_acc = np.mean([metrics[p]["accuracy"] for p in pathologies])

    metrics["macro"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "accuracy": macro_acc
    }
    return metrics


if __name__ == "__main__":
    generated = [
        "Normal heart size, mild edema, possible consolidation in the right base",
        "Large cardiac silhouette, no signs of edema, pleural effusion noted"
    ]
    reference = [
        "Heart is normal in size, no edema, pneumonia in right base",
        "Cardiomegaly present, no effusion or edema"
    ]

    results = evaluate_clinical_accuracy(generated, reference)
    print("Clinical Accuracy (CheXpert-based) = ", results)
