from kfp import dsl, compiler
import kfp
from kfp.dsl import Input, Output, Dataset, Model, ClassificationMetrics, Metrics, HTML


# Step 1: Load Dataset
@dsl.component(base_image="python:3.9", packages_to_install=["pandas", "scikit-learn"])
def load_data(output_csv: Output[Dataset]):
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df.to_csv(output_csv.path, index=False)


# Step 2: Preprocess Data
@dsl.component(base_image="python:3.9", packages_to_install=["pandas", "scikit-learn"])
def preprocess_data(
    input_csv: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    output_ytrain: Output[Dataset],
    output_ytest: Output[Dataset],
):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_csv.path)
    features = df.drop(columns=["target"])
    target = df["target"]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, target, test_size=0.2, random_state=42, stratify=target
    )

    pd.DataFrame(X_train, columns=features.columns).to_csv(
        output_train.path, index=False
    )
    pd.DataFrame(X_test, columns=features.columns).to_csv(output_test.path, index=False)
    pd.DataFrame({"target": y_train}).to_csv(output_ytrain.path, index=False)
    pd.DataFrame({"target": y_test}).to_csv(output_ytest.path, index=False)


# Step 3: Train Model
@dsl.component(
    base_image="python:3.9", packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    train_data: Input[Dataset],
    ytrain_data: Input[Dataset],
    model_output: Output[Model],
    epochs: int = 10,
):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from joblib import dump

    X_train = pd.read_csv(train_data.path)
    y_train = pd.read_csv(ytrain_data.path)["target"].to_numpy()

    model = LogisticRegression(max_iter=epochs, random_state=42)
    model.fit(X_train, y_train)

    dump(model, model_output.path)

    # Helpful metadata for artifact inspection
    model_output.metadata["framework"] = "scikit-learn"
    model_output.metadata["model_type"] = "LogisticRegression"
    model_output.metadata["max_iter"] = epochs

    print(f"Model trained for {epochs} iterations and saved.")


# Step 4: Evaluate Model + Visualizations
@dsl.component(
    base_image="python:3.9", packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate_model(
    test_data: Input[Dataset],
    ytest_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[ClassificationMetrics],
    scalar_metrics: Output[Metrics],
    html_report: Output[HTML],
):
    import pandas as pd
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    from joblib import load
    import html

    X_test = pd.read_csv(test_data.path)
    y_test = pd.read_csv(ytest_data.path)["target"].to_numpy()

    clf = load(model.path)
    y_pred = clf.predict(X_test)

    class_names = ["setosa", "versicolor", "virginica"]

    # Scalar metrics
    acc = accuracy_score(y_test, y_pred)
    scalar_metrics.log_metric("accuracy", float(acc))
    scalar_metrics.log_metric("num_test_samples", float(len(y_test)))

    # Kubeflow confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    metrics.log_confusion_matrix(categories=class_names, matrix=cm.tolist())

    # Add a few metadata fields for easier debugging
    metrics.metadata["accuracy"] = float(acc)
    metrics.metadata["classes"] = class_names

    # Build a reliable HTML artifact so the UI has a concrete file to render
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    html_content = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>Iris Evaluation Report</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            margin: 24px;
          }}
          h1, h2 {{
            color: #222;
          }}
          table {{
            border-collapse: collapse;
            margin: 12px 0 24px 0;
            width: auto;
          }}
          th, td {{
            border: 1px solid #ccc;
            padding: 8px 12px;
            text-align: center;
          }}
          th {{
            background: #f5f5f5;
          }}
          .metric {{
            font-size: 18px;
            margin-bottom: 16px;
          }}
          .note {{
            color: #666;
            font-size: 13px;
          }}
        </style>
      </head>
      <body>
        <h1>Iris Model Evaluation</h1>
        <div class="metric"><strong>Accuracy:</strong> {acc:.4f}</div>

        <h2>Confusion Matrix</h2>
        {cm_df.to_html(border=0)}

        <h2>Classification Report</h2>
        {report_df.to_html(border=0)}

        <p class="note">
          Rows in the confusion matrix are true labels; columns are predicted labels.
        </p>
      </body>
    </html>
    """

    with open(html_report.path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Accuracy: {acc:.4f}")


@dsl.pipeline(name="ml-pipeline-local")
def ml_pipeline(epochs: int = 100):
    load_op = load_data()

    preprocess_op = preprocess_data(input_csv=load_op.outputs["output_csv"])

    train_op = train_model(
        train_data=preprocess_op.outputs["output_train"],
        ytrain_data=preprocess_op.outputs["output_ytrain"],
        epochs=epochs,
    )

    evaluate_op = evaluate_model(
        test_data=preprocess_op.outputs["output_test"],
        ytest_data=preprocess_op.outputs["output_ytest"],
        model=train_op.outputs["model_output"],
    )

    # Optional: make caching explicit
    load_op.set_caching_options(True)
    preprocess_op.set_caching_options(True)
    train_op.set_caching_options(True)
    evaluate_op.set_caching_options(True)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline, package_path="pipeline3_local.yaml"
    )

    client = kfp.Client(host="http://localhost:8080")

    MY_EXPERIMENT_NAME = "Classification"

    client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={"epochs": 150},
        experiment_name=MY_EXPERIMENT_NAME,
        run_name="Run_150_epochs",
    )
