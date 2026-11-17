from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / 'Analysis.csv', sep=';')
print(data.head())

corr = data.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr, annot=True, mask=mask, fmt='.1g', cmap='coolwarm', ax=ax)
ax.set_title('Матрица корреляции признаков')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(data['X6'], data['Y1'], 'o', color='green')
ax.set_xlabel('X6')
ax.set_ylabel('Y1')
ax.set_title('Диаграмма рассеяния X6 и Y1')
fig.tight_layout()
plt.show()

print('Модель 1: Y1 ~ X6')
x1 = sm.add_constant(data['X6'])
y1 = data['Y1']
model1 = sm.OLS(y1, x1).fit()
print(model1.summary())

print('Модель 2: Y2 ~ X3 + X7 + X8')
x2 = sm.add_constant(data[['X3', 'X7', 'X8']])
y2 = data['Y2']
model2 = sm.OLS(y2, x2).fit()
print(model2.summary())

print('Модель 3: Y3 ~ X1 + X4 + X5 + X7 + X8')
x3 = sm.add_constant(data[['X1', 'X4', 'X5', 'X7', 'X8']])
y3 = data['Y3']
model3 = sm.OLS(y3, x3).fit()
print(model3.summary())

prediction = model3.predict(x3)
mse = mean_squared_error(y3, prediction)
print('MSE:', round(mse, 4), 'RMSE:', round(math.sqrt(mse), 4))

#Самостоятельная

X = data[['X1', 'X4', 'X5', 'X7', 'X8']]
y = data['Y3']
features = list(X.columns)

while len(features) > 1:
    model = sm.OLS(y, sm.add_constant(X[features])).fit()
    pvalues = model.pvalues.drop('const')
    if pvalues.max() <= 0.05:
        break
    features.remove(pvalues.idxmax())

print('Признаки шаговой регрессии:', features)
X_step = sm.add_constant(X[features])
step_model = sm.OLS(y, X_step).fit()
step_pred = step_model.predict(X_step)

ols_model = sm.OLS(y, sm.add_constant(X)).fit()
ols_pred = ols_model.predict(sm.add_constant(X))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 100)).fit(X_scaled, y)
ridge_pred = ridge_model.predict(X_scaled)

lasso_model = LassoCV(alphas=np.logspace(-3, 3, 100), cv=5, max_iter=10000).fit(X_scaled, y)
lasso_pred = lasso_model.predict(X_scaled)

results = {}
for name, pred in [('OLS', ols_pred), ('Stepwise', step_pred), ('Ridge', ridge_pred), ('Lasso', lasso_pred)]:
    mse = mean_squared_error(y, pred)
    results[name] = {'MSE': mse, 'RMSE': math.sqrt(mse)}

print(pd.DataFrame(results).T.round(4))
