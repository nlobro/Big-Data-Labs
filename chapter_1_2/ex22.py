from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.discrete.discrete_model import Probit, Logit

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / 'FinCond.csv', sep=';')
print(data.head())
print(data.describe())

Y = data['Y']
X = data[['Х1', 'Х2']]
X = sm.add_constant(X)

probit_model = Probit(Y, X.astype(float)).fit()
print(probit_model.summary())

mean_x1 = data['Х1'].mean()
mean_x2 = data['Х2'].mean()
b0 = probit_model.params['const']
b1 = probit_model.params['Х1']
b2 = probit_model.params['Х2']

z = b0 + b1 * mean_x1 + b2 * mean_x2
pb = norm.cdf(z)
print('z =', round(z, 4))
print('Вероятность банкротства средней организации:', round(pb, 4))

px1 = norm.pdf(z) * b1
px2 = norm.pdf(z) * b2
print('Предельный эффект Х1:', round(px1, 4))
print('Предельный эффект Х2:', round(px2, 4))

#Самостоятельная

logit_model = Logit(Y, X.astype(float)).fit()
print(logit_model.summary())

quality = pd.DataFrame({
    'Probit': [probit_model.prsquared, probit_model.aic, probit_model.bic],
    'Logit': [logit_model.prsquared, logit_model.aic, logit_model.bic]
}, index=['Pseudo R-squared', 'AIC', 'BIC'])
print(quality)
