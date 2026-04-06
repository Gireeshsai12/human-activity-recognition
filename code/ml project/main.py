import os
import glob
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = "."

ACCEL_PHONE_DIR = os.path.join(BASE_DIR, "accel_phone")
GYRO_PHONE_DIR  = os.path.join(BASE_DIR, "gyro_phone")
ACCEL_WATCH_DIR = os.path.join(BASE_DIR, "accel_watch")
GYRO_WATCH_DIR  = os.path.join(BASE_DIR, "gyro_watch")

def load_arff_file(path, sensor_type, device_type):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode("utf-8")
            except Exception:
                pass

    df["sensor_type"] = sensor_type
    df["device_type"] = device_type
    return df

def load_folder(folder_path, sensor_type, device_type):
    dfs = []
    for path in glob.glob(os.path.join(folder_path, "*.arff")):
        df = load_arff_file(path, sensor_type, device_type)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df_accel_phone = load_folder(ACCEL_PHONE_DIR, "accel", "phone")
df_gyro_phone  = load_folder(GYRO_PHONE_DIR,  "gyro",  "phone")
df_accel_watch = load_folder(ACCEL_WATCH_DIR, "accel", "watch")
df_gyro_watch  = load_folder(GYRO_WATCH_DIR,  "gyro",  "watch")

df_all = pd.concat(
    [df_accel_phone, df_gyro_phone, df_accel_watch, df_gyro_watch],
    ignore_index=True
)

print("Full combined shape:", df_all.shape)
print("Raw columns:", df_all.columns)

df_all.columns = [c.replace('"', '') for c in df_all.columns]
print("Cleaned columns:", df_all.columns)

label_col = "ACTIVITY"
print("Unique activities:", df_all[label_col].unique())


activity_counts = df_all[label_col].value_counts()
print("\nActivity counts:")
print(activity_counts)

top_10_activities = activity_counts.head(10).index.tolist()
print("\nTop 10 activities used for classification:")
print(top_10_activities)

df_filtered = df_all[df_all[label_col].isin(top_10_activities)].copy()
print("\nFiltered shape (top 10 activities):", df_filtered.shape)
print(df_filtered[label_col].value_counts())
cols_to_drop = [label_col]

if "class" in df_filtered.columns:
    cols_to_drop.append("class")

X = df_filtered.drop(columns=cols_to_drop)



if "sensor_type" in X.columns or "device_type" in X.columns:
    X = pd.get_dummies(X, columns=["sensor_type", "device_type"], drop_first=True)

y = df_filtered[label_col]

le = LabelEncoder()
y_enc = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, multi_class="multinomial"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM_RBF": SVC(kernel="rbf", C=10, gamma="scale"),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    print(f"\n===== Training {name} =====")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results[name] = (model, y_pred, acc)


best_name = max(results, key=lambda k: results[k][2])
best_model, best_pred, best_acc = results[best_name]

print("\nBest Model:", best_name, "Accuracy:", best_acc)

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.show()
