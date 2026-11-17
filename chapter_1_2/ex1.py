from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / 'Customers.csv', sep=';')
df = df[['Prices', 'Number', 'Customer_id']].astype(float)

print(df.head())
print(df.describe())
print(df.dtypes)

for col in df.columns:
    print(col, '-', round(df[col].isnull().mean() * 100, 4), '% пропусков')

median = df['Customer_id'].median()
df['Customer_id'] = df['Customer_id'].fillna(median)

original = df.copy()
medianP = df['Prices'].median()
df.loc[(df['Prices'] > 100000) | (df['Prices'] < 0), 'Prices'] = medianP

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df.loc[df['Prices'] < 2500, 'Prices'], bins=50, edgecolor='black', color='royalblue')
ax.set_title('Гистограмма распределения цен покупок')
ax.set_xlabel('Prices')
ax.set_ylabel('Частота')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
plot_data = df.sample(n=min(len(df), 20000), random_state=42)
ax.scatter(plot_data['Customer_id'], plot_data['Prices'], color='blue', s=3, alpha=0.25)
ax.set_title('Диаграмма рассеяния')
ax.set_xlabel('Customer_id')
ax.set_ylabel('Prices')
ax.grid(True)
fig.tight_layout()
plt.show()

#Самостоятельная

df1 = original[(original['Prices'] <= 100000) & (original['Prices'] >= 0)].copy()
print('Строк после удаления выбросов:', len(df1))

fig, ax = plt.subplots(figsize=(10, 5))
plot_data = df1.sample(n=min(len(df1), 20000), random_state=42)
ax.scatter(plot_data['Customer_id'], plot_data['Prices'], color='blue', s=3, alpha=0.25)
ax.set_title('Диаграмма рассеяния после удаления выбросов')
ax.set_xlabel('Customer_id')
ax.set_ylabel('Prices')
ax.grid(True)
fig.tight_layout()
plt.show()

print('Заказы дороже 10 млн руб.:')
print(original[original['Prices'] > 10000000])

df2 = original[original['Prices'] <= 100].copy()
print('Покупки стоимостью до 100 руб.:', len(df2))
print(df2.head())

df3 = original.copy()
df3['Prices'] = df3['Prices'] * 1.2
print('Цены после увеличения на 20%:')
print(df3.head())
