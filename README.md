# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
#### Date: 16.09.2025
#### Name: SHYAM S
#### Register.No: 212223240156

## AIM:
To implement ARMA model in python.
## ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
## PROGRAM:
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('/content/index_1.csv')

data['datetime'] = pd.to_datetime(data['datetime'])

ts = data.set_index('datetime').resample('H')['money'].sum()

N = 1000
plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(ts)
plt.title('Coffee Shop Revenue (Hourly)')
plt.ylabel("Money Spent")
plt.show()

plt.subplot(2, 1, 1)
plot_acf(ts, lags=int(len(ts) / 4), ax=plt.gca())
plt.title('Revenue ACF')

plt.subplot(2, 1, 2)
plot_pacf(ts, lags=int(len(ts) / 4), ax=plt.gca())
plt.title('Revenue PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(ts, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process (Coffee Revenue)')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title('ARMA(1,1) ACF (Simulated)')
plt.show()

plot_pacf(ARMA_1)
plt.title('ARMA(1,1) PACF (Simulated)')
plt.show()

arma22_model = ARIMA(ts, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process (Coffee Revenue)')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title('ARMA(2,2) ACF (Simulated)')
plt.show()

plot_pacf(ARMA_2)
plt.title('ARMA(2,2) PACF (Simulated)')
plt.show()
```
## OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/a51b3700-9b11-4377-8e8f-9b7b58834030" />

Partial Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/9c78cfca-d3ab-4c31-9a3c-3051be867442" />

Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/5ab4a461-72f1-4a5d-98db-5b420708faad" />

SIMULATED ARMA(2,2) PROCESS:
<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/818cf853-79db-4ba6-a730-dc1a39e71656" />

Partial Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/d154aadc-1768-409c-b905-024111f2368c" />

Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/1ea7fc88-ac89-4846-8b7c-c5a8e435cbeb" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
