from pathlib import Path
import gradio as gr
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

folder = Path(__file__).resolve().parent

iris = datasets.load_iris()
iris_model = RandomForestClassifier(random_state=42)
iris_model.fit(pd.DataFrame(iris.data, columns=iris.feature_names), iris.target)

X_reg, y_reg = datasets.make_regression(n_samples=1000, n_features=2, noise=20, random_state=42)
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    X = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                     columns=iris.feature_names)
    prediction = iris_model.predict(X)[0]
    probabilities = iris_model.predict_proba(X)[0]
    return iris.target_names[prediction], pd.DataFrame({
        'Вид': iris.target_names,
        'Вероятность': probabilities
    })


def predict_regression(x1, x2):
    return float(reg_model.predict(np.array([[x1, x2]]))[0])


#Самостоятельная

analysis = pd.read_csv(folder / 'Analysis.csv', sep=';')
X_ols = sm.add_constant(analysis[['X6']])
ols_model = sm.OLS(analysis['Y1'], X_ols).fit()

fincond = pd.read_csv(folder / 'FinCond.csv', sep=';')
X_probit = sm.add_constant(fincond[['Х1', 'Х2']])
probit_model = sm.Probit(fincond['Y'], X_probit).fit(disp=False)

clusters = pd.read_csv(folder / 'Clusters.csv', sep=';')
X_clusters = clusters.apply(pd.to_numeric)
kmeans_model = KMeans(n_clusters=5, random_state=111, n_init=10)
clusters['cluster'] = kmeans_model.fit_predict(X_clusters)
cluster_means = clusters.groupby('cluster').mean(numeric_only=True).reset_index()


def predict_ols(x6):
    x = pd.DataFrame({'const': [1.0], 'X6': [x6]})
    prediction = ols_model.predict(x)[0]
    return round(float(prediction), 2), round(float(ols_model.rsquared), 4)


def predict_probit(x1, x2):
    z = (probit_model.params['const'] +
         probit_model.params['Х1'] * x1 + probit_model.params['Х2'] * x2)
    return (round(float(norm.cdf(z)), 4),
            round(float(norm.pdf(z) * probit_model.params['Х1']), 4),
            round(float(norm.pdf(z) * probit_model.params['Х2']), 4))


def show_cluster(number):
    index = int(number) - 1
    return str(int(clusters.iloc[index]['cluster'])), clusters.iloc[[index]].copy()


with gr.Blocks(title='Лабораторная 3.2') as demo:
    gr.Markdown('# Лабораторная работа 3.2 — Gradio')

    with gr.Tab('Классификация ирисов'):
        gr.Markdown('Классификация ириса по параметрам чашелистика и лепестка')
        a = gr.Slider(4.3, 7.9, value=5.4, step=0.1, label='Sepal length')
        b = gr.Slider(2.0, 4.4, value=3.4, step=0.1, label='Sepal width')
        c = gr.Slider(1.0, 6.9, value=1.3, step=0.1, label='Petal length')
        d = gr.Slider(0.1, 2.5, value=0.2, step=0.1, label='Petal width')
        result = gr.Textbox(label='Прогноз')
        probabilities = gr.Dataframe(label='Вероятность по видам')
        gr.Button('Классифицировать').click(predict_iris, [a, b, c, d], [result, probabilities])

    with gr.Tab('Пример регрессии'):
        gr.Markdown('Линейная регрессия по искусственно созданным данным, как в методичке')
        r1 = gr.Number(value=1, label='Переменная x1')
        r2 = gr.Number(value=3, label='Переменная x2')
        result_reg = gr.Number(label='Предсказанное значение y')
        gr.Button('Предсказать').click(predict_regression, [r1, r2], result_reg)

    with gr.Tab('OLS: Analysis'):
        gr.Markdown('Прогноз Y1 по фактору X6')
        x6 = gr.Number(value=float(analysis['X6'].median()), label='X6')
        pred_ols = gr.Number(label='Прогноз Y1')
        r_squared = gr.Number(label='R-squared')
        gr.Button('Рассчитать').click(predict_ols, x6, [pred_ols, r_squared])

    with gr.Tab('Probit: FinCond'):
        gr.Markdown('Вероятность банкротства организации')
        p1 = gr.Number(value=float(fincond['Х1'].mean()), label='Х1')
        p2 = gr.Number(value=float(fincond['Х2'].mean()), label='Х2')
        p = gr.Number(label='Вероятность банкротства (0–1)')
        eff1 = gr.Number(label='Предельный эффект Х1')
        eff2 = gr.Number(label='Предельный эффект Х2')
        gr.Button('Рассчитать').click(predict_probit, [p1, p2], [p, eff1, eff2])

    with gr.Tab('KMeans: Clusters'):
        gr.Markdown('Распределение регионов между пятью кластерами')
        n = gr.Slider(1, len(clusters), value=1, step=1, label='Номер региона')
        label = gr.Textbox(label='Номер кластера')
        region = gr.Dataframe(label='Данные региона')
        gr.Button('Показать кластер').click(show_cluster, n, [label, region])
        gr.Dataframe(value=cluster_means, label='Средние значения по кластерам')


if __name__ == '__main__':
    demo.launch()