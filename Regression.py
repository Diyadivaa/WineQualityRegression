import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
data = pd.read_csv(url, sep=';')


X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Mean score: {scores.mean():.2f}')
print(f'Standard deviation: {scores.std():.2f}')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error


r2 = r2_score(y_test, y_pred)
print(f'R squared: {r2:.2f}')


mae = mean_absolute_error(y_test, y_pred)
print(f'Mean absolute error: {mae:.2f}')
