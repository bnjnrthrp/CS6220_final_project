# pip install pyarrow fastparquet

from config import SCRIPT_DIR, OUTPUT_DIR
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


'''
Expected column format of csv file is:
Date, Ticker, Adj Close, Ret_1d, Ret_5d, Ret_20d, Ret_60d, Ret_120d, Vol_10d, Vol_20d, Vol_60d, MACD, MACD_Signal, RSI14

Date is there just to organize the df in chronological order.
Target is taken from Ret_1d, which will become 1 for positive returns and 0 for negative returns.
'''

def main():
    drop_cols = ['label', 'Ticker', 'Date', 'Ret_1d']

    # Load data and finalize the preparations for PCA
    path = os.path.join(SCRIPT_DIR, './data/sp500_features.csv')
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.sort_values('Date')
    original_len = len(df)

    # Create the targets based off the 1 day return
    # Multi-class labels
    df['label'] = 1  # default = small/no move
    df.loc[df['Ret_1d'] > 0.015, 'label'] = 2  # big positive move
    df.loc[df['Ret_1d'] < -0.015, 'label'] = 0  # big negative move

    print(f"Label distribution:\n{df['label'].value_counts(normalize=True)}")    


    # Separate the feature set and the target, prepare for PCA analysis
    X = df.drop(columns=drop_cols)
    y = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Don't know the correct number of components, so instead capture 95% of the variance.
    pca = PCA(n_components=0.95)  # keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)

    # Save PCA-transformed features
    df_pca = pd.DataFrame(X_pca)
    df_pca['label'] = y

    out_path = os.path.join(OUTPUT_DIR, "pca_features.parquet")
    df_pca.to_parquet(out_path, index=False)

    # Split the data up, but maintain chronological order
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, shuffle=False  # Time-series style split
    )

    # Establish the logistic baseline
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Print the results
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }

    print("====== Logistic Regression Baseline Metrics: ======")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")

    print(classification_report(y_test, y_pred, target_names=['down', 'neutral', 'up']))

    return

# Allow running as script or import
if __name__ == "__main__":
    main()
