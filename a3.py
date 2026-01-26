import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv("lab_2.csv")
df.columns = df.columns.str.strip()
df['Price'] = df['Price'].str.replace(',','').astype(float)
df['Chg%'] = df['Chg%'].str.replace('%','').astype(float)
df = df.dropna(subset=['Day','Chg%'])

prices = df['Price'].values
chg = df['Chg%'].values
days = df['Day'].astype(str).values
months = df['Month'].astype(str).values

def m_mean(arr): return sum(arr)/len(arr)
def m_var(arr): mu=m_mean(arr); return sum((x-mu)**2 for x in arr)/len(arr)

mean_np = np.mean(prices)
var_np = np.var(prices)
mean_manual = m_mean(prices)
var_manual = m_var(prices)

times_numpy, times_manual = [], []
for _ in range(10):
    start=time.time(); np.mean(prices); np.var(prices); times_numpy.append(time.time()-start)
    start=time.time(); m_mean(prices); m_var(prices); times_manual.append(time.time()-start)
avg_time_numpy = sum(times_numpy)/10
avg_time_manual = sum(times_manual)/10

wed_prices = prices[days=='Wed']
wed_mean = m_mean(wed_prices) if len(wed_prices)>0 else 0
apr_prices = prices[months=='Apr']
apr_mean = m_mean(apr_prices) if len(apr_prices)>0 else 0

prob_loss = len([x for x in chg if x<0])/len(chg)
wed_chg = chg[days=='Wed']
prob_profit_wed = len([x for x in wed_chg if x>0])/len(wed_chg) if len(wed_chg)>0 else 0
cond_prob = prob_profit_wed

plt.scatter(days, chg)
plt.xlabel("Day of Week")
plt.ylabel("Chg%")
plt.show()

print(mean_np, var_np)
print(mean_manual, var_manual)
print(avg_time_numpy, avg_time_manual)
print(wed_mean, apr_mean)
print(prob_loss, prob_profit_wed, cond_prob)
