from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor, LazyClassifier

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / 'Rural_prices.csv', sep=';')

print(data.head())
print(data.columns.tolist())
print(data.describe())

X = data.drop('sale price', axis=1)
y = data['sale price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

reg_plot = models.reset_index().rename(columns={'index': 'Model'})
reg_plot = reg_plot.rename(columns={reg_plot.columns[0]: 'Model'})
height = max(8, len(reg_plot) * 0.35)

fig, ax = plt.subplots(figsize=(11, height))
sns.barplot(data=reg_plot, y='Model', x='R-Squared', color='steelblue', ax=ax)
ax.set_title('R-Squared моделей регрессии')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(11, height))
sns.barplot(data=reg_plot, y='Model', x='RMSE', color='steelblue', ax=ax)
ax.set_title('RMSE моделей регрессии')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(11, height))
sns.barplot(data=reg_plot, y='Model', x='Time Taken', color='steelblue', ax=ax)
ax.set_title('Время работы моделей регрессии')
fig.tight_layout()
plt.show()

print('Лучшая регрессия по RMSE:', models['RMSE'].idxmin())

#Самостоятельная

data = pd.read_csv(folder / 'FinCond.csv', sep=';')
X = data[['Х1', 'Х2']]
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

clf_plot = models.reset_index().rename(columns={'index': 'Model'})
clf_plot = clf_plot.rename(columns={clf_plot.columns[0]: 'Model'})
height = max(8, len(clf_plot) * 0.35)

fig, ax = plt.subplots(figsize=(11, height))
sns.barplot(data=clf_plot, y='Model', x='Accuracy', color='steelblue', ax=ax)
ax.set_title('Accuracy моделей классификации')
ax.set_xlim(0, 1)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(11, height))
sns.barplot(data=clf_plot, y='Model', x='F1 Score', color='steelblue', ax=ax)
ax.set_title('F1 Score моделей классификации')
ax.set_xlim(0, 1)
fig.tight_layout()
plt.show()

print('Лучшая классификация по Accuracy:', models['Accuracy'].idxmax())
