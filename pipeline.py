from typing import Dict, List
from kfp import dsl
from kfp import compiler
import kfp
from kfp.dsl import Input, Output, Dataset, Model, ClassificationMetrics, Metrics, HTML
import os


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
        scaled_features, target, test_size=0.2, random_state=42
    )

    pd.DataFrame(X_train, columns=features.columns).to_csv(
        output_train.path, index=False
    )
    pd.DataFrame(X_test, columns=features.columns).to_csv(output_test.path, index=False)
    pd.DataFrame(y_train).to_csv(output_ytrain.path, index=False)
    pd.DataFrame(y_test).to_csv(output_ytest.path, index=False)


# Step 3: Train Model (Added Epochs Parameter)
@dsl.component(
    base_image="python:3.9", packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    train_data: Input[Dataset],
    ytrain_data: Input[Dataset],
    model_output: Output[Model],
    epochs: int = 10,  # Added parameter
):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from joblib import dump

    X_train = pd.read_csv(train_data.path)
    y_train = pd.read_csv(ytrain_data.path).values.ravel()

    # Note: LogisticRegression uses 'max_iter' instead of 'epochs'
    model = LogisticRegression(max_iter=epochs)
    model.fit(X_train, y_train)

    dump(model, model_output.path)
    print(f"Model trained for {epochs} iterations and saved.")


# Step 4: Evaluate Model (Added Visualizations)
@dsl.component(
    base_image="python:3.9", packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate_model(
    test_data: Input[Dataset],
    ytest_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[ClassificationMetrics],
    scalar_metrics: Output[Metrics],
):
    import pandas as pd
    from sklearn.metrics import confusion_matrix, accuracy_score
    from joblib import load

    X_test = pd.read_csv(test_data.path)
    y_test = pd.read_csv(ytest_data.path)

    clf = load(model.path)
    y_pred = clf.predict(X_test)

    # 1. Log Scalar Metrics (Accuracy)
    acc = accuracy_score(y_test, y_pred)
    scalar_metrics.log_metric("accuracy", float(acc))

    # 2. Log Confusion Matrix (Visualizes in Kubeflow UI)
    cm = confusion_matrix(y_test, y_pred)
    metrics.log_confusion_matrix(
        categories=["setosa", "versicolor", "virginica"], matrix=cm.tolist()
    )


# Define the pipeline with configurable arguments
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


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline, package_path="pipeline3_local.yaml"
    )

    client = kfp.Client(host="http://localhost:8080")

    # Define the experiment name here
    MY_EXPERIMENT_NAME = "Classification"

    client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={"epochs": 150},
        experiment_name=MY_EXPERIMENT_NAME,  # This sets the experiment name
        run_name="Run_150_epochs",  # Optional: sets the specific run name
    )
