import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

def generate_dataset(n, seed=0):
    np.random.seed(seed)
    df = pd.DataFrame({
        'Income': np.random.randint(0, 100001, size=n),
        'Gender': np.random.choice([0, 1], size=n)
    })
    return df

def xor_label(df):
    return ((df['Income'] > 50000).astype(int) ^ df['Gender']).rename('Label')

def main():
    df = generate_dataset(500, seed=42)
    df['XOR_Label'] = xor_label(df)
    df.to_csv('xor_data.csv', index=False)

    train_linear_regression(df)

def train_linear_regression(df):
    X = df[['Income', 'Gender']].values
    y = df['XOR_Label'].values

    model = LogisticRegression(max_iter=1000).fit(X, y)
    print(f'Accuracy: {model.score(X, y):.2f}')

    w0, w1 = model.coef_[0]
    b = model.intercept_[0]

    print(f'Weights: w0 = {w0:.4f}, w1 = {w1:.4f}, bias = {b:.4f}')

if __name__ == "__main__":
    main()