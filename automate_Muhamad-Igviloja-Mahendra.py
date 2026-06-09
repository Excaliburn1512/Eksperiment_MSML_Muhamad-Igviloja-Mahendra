import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import argparse


def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus kolom yang tidak relevan (identifier)."""
    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[INFO] Kolom dihapus: {existing}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding fitur kategorikal:
    - LabelEncoding untuk Gender (biner)
    - OneHotEncoding untuk Geography dan Card Type
    """
    # Label Encoding: Gender
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        print(f"[INFO] Label Encoding: Gender -> {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # One-Hot Encoding: Geography, Card Type
    ohe_cols = [c for c in ['Geography', 'Card Type'] if c in df.columns]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
        print(f"[INFO] One-Hot Encoding diterapkan pada: {ohe_cols}")

    return df


def handle_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR Capping pada fitur kontinu.
    Nilai di luar [Q1 - 1.5*IQR, Q3 + 1.5*IQR] di-clip ke batas tersebut.
    """
    outlier_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Point Earned']
    outlier_cols = [c for c in outlier_cols if c in df.columns]

    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"[INFO] Outlier capping {col}: {before} outlier -> range [{lower:.2f}, {upper:.2f}]")

    return df


def scale_features(df: pd.DataFrame, target_col: str = 'Exited') -> pd.DataFrame:
    """
    Normalisasi fitur numerik kontinu menggunakan StandardScaler.
    Kolom target tidak ikut di-scale.
    """
    scale_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary',
                  'Satisfaction Score', 'Point Earned']
    scale_cols = [c for c in scale_cols if c in df.columns and c != target_col]

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"[INFO] StandardScaler diterapkan pada: {scale_cols}")
    return df


def split_data(df: pd.DataFrame, target_col: str = 'Exited', test_size: float = 0.2,
               random_state: int = 42):
    """
    Membagi data menjadi train dan test set dengan stratified split.
    Returns: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train-Test Split (80:20)")
    print(f"       X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"       y_train dist: {y_train.value_counts().to_dict()}")
    print(f"       y_test dist:  {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def preprocess(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Pipeline preprocessing lengkap dari raw data hingga siap dilatih.

    Parameters
    ----------
    input_path : str
        Path ke file CSV raw dataset.
    output_path : str, optional
        Path untuk menyimpan hasil preprocessing. Jika None, tidak disimpan.

    Returns
    -------
    pd.DataFrame
        Dataset yang telah diproses dan siap dilatih.
    """
    print("=" * 55)
    print("  AUTOMATE PREPROCESSING - Customer Churn Records")
    print("=" * 55)

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Drop kolom tidak relevan
    df = drop_irrelevant_columns(df)

    # Step 3: Tidak ada missing values (validasi)
    missing = df.isnull().sum().sum()
    print(f"[INFO] Missing values: {missing}")

    # Step 4: Duplikat
    dup = df.duplicated().sum()
    print(f"[INFO] Duplikat dihapus: {dup}")
    if dup > 0:
        df = df.drop_duplicates()

    # Step 5: Encoding kategorikal
    df = encode_categorical(df)

    # Step 6: Handling outlier
    df = handle_outliers_iqr(df)

    # Step 7: Feature scaling
    df = scale_features(df, target_col='Exited')

    # Step 8: Simpan hasil
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[INFO] Dataset preprocessing disimpan ke: {output_path}")

    print(f"\n[DONE] Shape akhir: {df.shape}")
    print("=" * 55)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate preprocessing Customer Churn Records")
    parser.add_argument(
        "--input", type=str,
        default="Customer-Churn-Records_raw.csv",
        help="Path ke file CSV raw (default: Customer-Churn-Records_raw.csv)"
    )
    parser.add_argument(
        "--output", type=str,
        default="Customer-Churn-Records_preprocessing.csv",
        help="Path output file CSV preprocessing (default: Customer-Churn-Records_preprocessing.csv)"
    )
    args = parser.parse_args()

    df_result = preprocess(input_path=args.input, output_path=args.output)
    print(f"\nPreview 5 baris pertama:")
    print(df_result.head())
