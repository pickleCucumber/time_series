import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import datetime
from dateutil.relativedelta import *
from itertools import product

def invboxcox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))


df = pd.read_csv('ICAAP_daily_credit.csv', sep=';',encoding='PT154')
dc=df.melt(id_vars=['date'])
dc['date']=pd.to_datetime(dc['variable'], format='%d.%m.%Y', errors='coerce')
del dc['variable']
dc['value'] = dc['value'].astype(str)
dc['value']=dc['value'].apply(lambda x:x.replace('%', ''))
dc['value'] = dc['value'].astype(float)
dc['date'] =pd.to_datetime(dc['date'], format='%Y-%m', errors='coerce')
dc['date'] = dc['date'].dt.strftime('%Y-%m')
dc['value'] = dc['value'].astype(float)
avg=dc.groupby([dc['date']], sort=False).mean()
print(avg)
#plt.figsize(15,7)
avg.plot()
#plt.show()
"""""

avg['value'], lmbda = stats.boxcox(avg.value)
#plt.figure(figsize(15,7))
avg['value'].plot()
#plt.show()

print("Оптимальный параметр преобразования Бокса-Кокса: %f" % lmbda)
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(avg['value'].values)[1])
"""""
avg['value_box_diff'] = avg.value - avg.value.shift(12)

avg.value_box_diff[12:].plot()
#plt.show()
#sm.tsa.seasonal_decompose(data.salary_box_diff[12:]).plot()
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(avg.value_box_diff[12:])[1])

avg['value_box_diff2'] = avg.value_box_diff - avg.value_box_diff.shift(1)

avg.value_box_diff2[13:].plot()
#plt.show()
#sm.tsa.seasonal_decompose(data.salary_box_diff[12:]).plot()
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(avg.value_box_diff2[13:])[1])


#Подбираем параметры
#plt.figure(figsize(15,8))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(avg.value_box_diff2[13:].values.squeeze(), lags=8, ax=ax)
#plt.show()
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(avg.value_box_diff2[13:].values.squeeze(), lags=8, ax=ax)
#plt.show()

#Рассмотрим следующие параметры: Q = 0, q = 1, P = 1, p = 5

ps = range(0, 6)
d=1
qs = range(0, 3)
Ps = range(0, 2)
D=1
Qs = range(0, 1)
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
#parameters[0]

#%%time
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')

for param in parameters_list:
    #try except нужен, потому что на некоторых наборах параметров модель не обучается
    try:
        model=sm.tsa.statespace.SARIMAX(avg.value, order=(param[0], d, param[1]),
                                        seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    #выводим параметры, на которых модель не обучается и переходим к следующему набору
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    #сохраняем лучшую модель, aic, параметры
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

warnings.filterwarnings('default')

result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
result_table.sort_values(by = 'aic', ascending=True).head()
print(result_table.sort_values(by = 'aic', ascending=True).head())
best_model.summary()
print(best_model.summary())

#остатки
#plt.figure(figsize(15,8))
plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')
plt.show()
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=17, ax=ax)
plt.show()
print("Критерий Стьюдента: p=%f" % stats.ttest_1samp(best_model.resid[13:], 0)[1])
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])



#Остатки несмещены (подтверждается критерием Стьюдента) стационарны (подтверждается критерием Дики-Фуллера и визуально)
#Посмотрим на итоговый результат
avg['model'] = best_model.fittedvalues

avg['value'].plot()
avg['model'].plot(color='r')

plt.show()
#print data.model[13:]

df2 = avg[['value']]
date_list = [datetime.datetime.strptime("2021-06", "%Y-%m") + relativedelta(months=x) for x in range(0,50)]
future = pd.DataFrame(index=date_list, columns = df2.columns)
df2 = pd.concat([df2, future])
"""""
df2['forecast'] = best_model.predict(start=41, end=60)

df2['value'].plot(color = 'g')
df2['forecast'].plot(color='r')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df2)

plt.show()
"""""
forecast = best_model.predict(start = len(avg),
                           end=len(avg)+24,
                           typ='levels')
#df['passengers'].plot(figsize=(12,8),legend=True)
#print(forecast)
df2 = pd.concat([avg['value'], forecast])
df2.plot()
plt.show()
print(df2)
