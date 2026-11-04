
import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def main(data_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset
    df = pd.read_csv(data_path)

    # 2. Clean dataset
    df['TotalCharges'] = df['TotalCharges'].replace(
        r'^\s*$', pd.NA, regex=True
    )
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    duplicate_count = df.duplicated().sum()
    print("Exact duplicate rows:", duplicate_count)
    df = df.drop_duplicates()

    # 3. Separate target and features
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop(
        columns=['Churn', 'customerID'],
        errors='ignore'
    )

    # 4. Identify numerical and categorical columns
    numerical_features = X.select_dtypes(
        include=['int64', 'float64']
    ).columns

    categorical_features = X.select_dtypes(
        include=['object']
    ).columns

    # 5. Build preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # 6. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 7. Define models
    models = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),

        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ])
    }

    # 8. Cross-validation
    scoring = {
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = {}

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring
        )

        results[name] = {
            metric: scores[f'test_{metric}'].mean()
            for metric in scoring
        }

    results_df = pd.DataFrame(results).T
    results_df.to_csv(
        os.path.join(output_dir, 'model_comparison.csv')
    )

    print("\nCross-validation results:")
    print(results_df)

    # 9. Select best model using ROC-AUC
    best_model_name = results_df['roc_auc'].idxmax()
    best_model = models[best_model_name]

    best_model.fit(X_train, y_train)

    print("\nSelected model:", best_model_name)

    # 10. Final test evaluation
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }

    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=['Metric', 'Score']
    )

    metrics_df.to_csv(
        os.path.join(output_dir, 'model_evaluation_metrics.csv'),
        index=False
    )

    print("\nFinal test metrics:")
    print(metrics_df)

    # 11. Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['No Churn', 'Churn']
    )

    display.plot(cmap='Blues')
    plt.title(f'{best_model_name} Confusion Matrix')
    plt.savefig(
        os.path.join(output_dir, 'confusion_matrix.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # 12. Save model comparison chart
    results_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'model_comparison.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # 13. Save Random Forest feature importance
    rf_model = models['Random Forest']
    rf_model.fit(X_train, y_train)

    rf = rf_model.named_steps['classifier']
    feature_names = (
        rf_model.named_steps['preprocessor'].get_feature_names_out()
    )

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    importance_df.to_csv(
        os.path.join(output_dir, 'feature_importance.csv'),
        index=False
    )

    top_features = importance_df.head(15).sort_values('Importance')

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features['Feature'],
        top_features['Importance']
    )
    plt.title('Top 15 Churn Prediction Features')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'feature_importance.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # 14. Save selected trained model
    joblib.dump(
        best_model,
        os.path.join(output_dir, 'churn_prediction_model.pkl')
    )

    print("\nTraining completed successfully!")
    print("Results saved in:", output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a customer churn prediction model'
    )

    parser.add_argument(
        '--data',
        default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )

    parser.add_argument(
        '--output',
        default='results'
    )

    args = parser.parse_args()

    main(args.data, args.output)
