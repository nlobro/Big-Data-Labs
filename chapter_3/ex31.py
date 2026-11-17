from pathlib import Path
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from scipy.stats import norm
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

folder = Path(__file__).resolve().parent
st.set_page_config(page_title='Лабораторная 3.1', layout='wide')


def iris_page():
    st.title('Классификация цветов')
    st.write('Приложение классифицирует вид ириса по выбранным параметрам.')
    st.sidebar.header('Параметры для классификации')

    iris = datasets.load_iris()
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)

    df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                      columns=iris.feature_names)
    st.subheader('Выбранные параметры пользователем')
    st.dataframe(df)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(pd.DataFrame(iris.data, columns=iris.feature_names), iris.target)
    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)

    st.subheader('Возможная классификация и индекс вида')
    st.write(list(enumerate(iris.target_names)))
    st.subheader('Прогноз')
    st.write(iris.target_names[prediction[0]])
    st.subheader('Вероятность прогноза')
    st.dataframe(pd.DataFrame(prediction_proba, columns=iris.target_names))


#Самостоятельная

def regression_page():
    st.title('Линейная регрессия OLS')
    data = pd.read_csv(folder / 'Analysis.csv', sep=';')
    X = sm.add_constant(data[['X6']])
    model = sm.OLS(data['Y1'], X).fit()
    x6 = st.number_input('Значение X6', value=float(data['X6'].median()))
    prediction = model.predict(pd.DataFrame({'const': [1.0], 'X6': [x6]}))[0]
    st.metric('Прогноз Y1', f'{prediction:.2f}')
    st.write('R-squared:', round(model.rsquared, 4))
    st.scatter_chart(data, x='X6', y='Y1')
    st.dataframe(pd.DataFrame({'Коэффициенты': model.params}))


def probit_page():
    st.title('Probit-регрессия')
    data = pd.read_csv(folder / 'FinCond.csv', sep=';')
    X = sm.add_constant(data[['Х1', 'Х2']])
    model = sm.Probit(data['Y'], X).fit(disp=False)
    x1 = st.number_input('Х1', value=float(data['Х1'].mean()))
    x2 = st.number_input('Х2', value=float(data['Х2'].mean()))
    z = model.params['const'] + model.params['Х1'] * x1 + model.params['Х2'] * x2
    probability = norm.cdf(z)
    st.metric('Вероятность банкротства', f'{probability:.2%}')
    st.write('Pseudo R-squared:', round(model.prsquared, 4))
    st.write('Предельный эффект Х1:', round(norm.pdf(z) * model.params['Х1'], 4))
    st.write('Предельный эффект Х2:', round(norm.pdf(z) * model.params['Х2'], 4))
    st.dataframe(pd.DataFrame({'Коэффициенты': model.params}))


def clusters_page():
    st.title('Кластерный анализ KMeans')
    data = pd.read_csv(folder / 'Clusters.csv', sep=';')
    X = data.apply(pd.to_numeric)
    model = KMeans(n_clusters=5, random_state=111, n_init=10)
    data['cluster'] = model.fit_predict(X)
    number = st.selectbox('Номер региона в таблице', range(1, len(data) + 1))
    st.write('Кластер региона:', int(data.iloc[number - 1]['cluster']))
    st.subheader('Количество регионов по кластерам')
    st.bar_chart(data['cluster'].value_counts().sort_index())
    st.subheader('Средние значения по кластерам')
    st.dataframe(data.groupby('cluster').mean(numeric_only=True))
    st.subheader('Распределение регионов')
    data['cluster'] = data['cluster'].astype(str)
    st.scatter_chart(data, x='X11', y='X12', color='cluster')


pg = st.navigation([
    st.Page(iris_page, title='Классификация ирисов', default=True),
    st.Page(regression_page, title='Линейная регрессия'),
    st.Page(probit_page, title='Probit-регрессия'),
    st.Page(clusters_page, title='KMeans-кластеризация')
])
pg.run()