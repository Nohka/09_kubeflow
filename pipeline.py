from kfp import dsl, compiler
import kfp
from kfp.dsl import Input, Output, Dataset, Model, ClassificationMetrics, Metrics, HTML


@dsl.component(base_image="python:3.9", packages_to_install=["pandas", "scikit-learn"])
def load_data(output_csv: Output[Dataset]):
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df.to_csv(output_csv.path, index=False)


@dsl.component(base_image="python:3.9", packages_to_install=["pandas", "scikit-learn"])
def preprocess_data(
    input_csv: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    output_ytrain: Output[Dataset],
    output_ytest: Output[Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(input_csv.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pd.DataFrame(X_train, columns=X.columns).to_csv(output_train.path, index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv(output_test.path, index=False)
    pd.DataFrame({"target": y_train}).to_csv(output_ytrain.path, index=False)
    pd.DataFrame({"target": y_test}).to_csv(output_ytest.path, index=False)


@dsl.component(
    base_image="python:3.9", packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    train_data: Input[Dataset],
    ytrain_data: Input[Dataset],
    model_output: Output[Model],
    epochs: int = 150,
):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from joblib import dump

    X_train = pd.read_csv(train_data.path)
    y_train = pd.read_csv(ytrain_data.path)["target"].to_numpy()

    model = LogisticRegression(
        max_iter=epochs,
        random_state=42,
        multi_class="auto",
    )
    model.fit(X_train, y_train)

    dump(model, model_output.path)
    model_output.metadata["framework"] = "scikit-learn"
    model_output.metadata["model_type"] = "LogisticRegression"
    model_output.metadata["max_iter"] = epochs


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib", "matplotlib"],
)
def evaluate_model(
    test_data: Input[Dataset],
    ytest_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[ClassificationMetrics],
    scalar_metrics: Output[Metrics],
    html_report: Output[HTML],
):
    import base64
    import io
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from joblib import load
    from sklearn.metrics import (
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        roc_curve,
    )
    from sklearn.preprocessing import label_binarize

    X_test = pd.read_csv(test_data.path)
    y_test = pd.read_csv(ytest_data.path)["target"].to_numpy()

    clf = load(model.path)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    class_names = ["setosa", "versicolor", "virginica"]
    class_ids = [0, 1, 2]

    acc = accuracy_score(y_test, y_pred)
    scalar_metrics.log_metric("accuracy", float(acc))
    scalar_metrics.log_metric("num_test_samples", float(len(y_test)))

    cm = confusion_matrix(y_test, y_pred, labels=class_ids)
    metrics.log_confusion_matrix(categories=class_names, matrix=cm.tolist())

    y_test_bin = label_binarize(y_test, classes=class_ids)

    # Per-class AUC metrics for easy comparison in the UI
    roc_rows = []
    for i, class_name in enumerate(class_names):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        auc_i = auc(fpr_i, tpr_i)
        scalar_metrics.log_metric(f"auc_{class_name}", float(auc_i))
        roc_rows.append({"class": class_name, "auc": round(float(auc_i), 4)})

    # Single ROC curve logged to Kubeflow UI: micro-average over all classes
    fpr_micro, tpr_micro, thresholds_micro = roc_curve(
        y_test_bin.ravel(),
        y_prob.ravel(),
    )

    finite_mask = np.isfinite(thresholds_micro)
    fpr_micro = fpr_micro[finite_mask]
    tpr_micro = tpr_micro[finite_mask]
    thresholds_micro = thresholds_micro[finite_mask]

    if len(fpr_micro) > 1:
        metrics.log_roc_curve(
            fpr=fpr_micro.tolist(),
            tpr=tpr_micro.tolist(),
            threshold=thresholds_micro.tolist(),
        )

    # Build an embedded ROC chart inside the HTML report
    plt.figure(figsize=(6, 5))
    for i, class_name in enumerate(class_names):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        auc_i = auc(fpr_i, tpr_i)
        plt.plot(fpr_i, tpr_i, label=f"{class_name} (AUC={auc_i:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    roc_plot_b64 = base64.b64encode(buf.read()).decode("utf-8")

    report_df = (
        pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                target_names=class_names,
                output_dict=True,
            )
        )
        .transpose()
        .round(4)
    )

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    roc_df = pd.DataFrame(roc_rows)

    html_content = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>Iris Evaluation Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          h1, h2 {{ color: #222; }}
          table {{ border-collapse: collapse; margin: 12px 0 24px 0; }}
          th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
          th {{ background: #f5f5f5; }}
          .metric {{ font-size: 18px; margin-bottom: 16px; }}
          img {{ max-width: 700px; height: auto; border: 1px solid #ddd; }}
        </style>
      </head>
      <body>
        <h1>Iris Model Evaluation</h1>
        <div class="metric"><strong>Accuracy:</strong> {acc:.4f}</div>

        <h2>Confusion Matrix</h2>
        {cm_df.to_html(border=0)}

        <h2>Per-class ROC AUC</h2>
        {roc_df.to_html(index=False, border=0)}

        <h2>ROC Plot</h2>
        <img src="data:image/png;base64,{roc_plot_b64}" alt="ROC plot" />

        <h2>Classification Report</h2>
        {report_df.to_html(border=0)}
      </body>
    </html>
    """

    with open(html_report.path, "w", encoding="utf-8") as f:
        f.write(html_content)


@dsl.pipeline(name="ml-pipeline-local")
def ml_pipeline(epochs: int = 152):
    load_op = load_data()
    preprocess_op = preprocess_data(input_csv=load_op.outputs["output_csv"])
    train_op = train_model(
        train_data=preprocess_op.outputs["output_train"],
        ytrain_data=preprocess_op.outputs["output_ytrain"],
        epochs=epochs,
    )
    evaluate_model(
        test_data=preprocess_op.outputs["output_test"],
        ytest_data=preprocess_op.outputs["output_ytest"],
        model=train_op.outputs["model_output"],
    )

    load_op.set_caching_options(False)
    preprocess_op.set_caching_options(False)
    train_op.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path="pipeline3_local.yaml",
    )

    client = kfp.Client(host="http://localhost:8080")
    client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={"epochs": 152},
        experiment_name="Classification",
        run_name="Run_152_epochs_fixed_roc",
    )
