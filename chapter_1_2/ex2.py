from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / 'Customers.csv', sep=';')
df = df[['Prices', 'Number', 'Customer_id']].astype(float)

print(df.head())
print(df.describe())

df_range = df[(df['Prices'] >= 0) & (df['Prices'] < 2500)]

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df_range['Prices'], bins=50, kde=True, color='blue', ax=ax)
ax.set_title('Цены покупок в интернет-магазине')
ax.set_xlabel('Prices')
ax.set_ylabel('Частота')
fig.tight_layout()
plt.show()

df = df.head(100).copy()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['Number'], df['Prices'], color='tomato')
ax.set_xlabel('Number')
ax.set_ylabel('Prices')
ax.set_title('Величина покупок для первой сотни покупателей')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
autocorrelation_plot(df['Prices'], ax=ax)
ax.set_title('Автокорреляционная функция')
fig.tight_layout()
plt.show()

#Самостоятельная

values = pd.DataFrame(df['Prices'].values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't']
result = dataframe.corr()
print('Матрица корреляции с лагом:')
print(result)

fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(df['Prices'], lags=25, ax=ax)
ax.set_title('Столбиковая автокорреляция')
fig.tight_layout()
plt.show()
