#2. Data Understanding

import pandas as pd

#2.1 Data Sources
transactions = pd.read_csv('../datasets/PSP_Jan_Feb_2019.csv')
service_fees = pd.read_csv('../datasets/Service_fees.csv')

#2.2 Describe data
transactions.head()
print(transactions.shape)

#2.3 Verify data quality
print(transactions.info())
print('There are:', transactions[transactions.duplicated()==True].shape[0],'data duplicate.')

#2.4 Explore data
import seaborn as sns
import matplotlib.pyplot as plt

Palette = ["#ff6961", "#a9d39e",'#fff44f','#74bbfb'] #define your preference
sns.set_style("whitegrid")
sns.set_palette(Palette) #use the list defined in the function

#2.4.1.1 Transactions and success rate by Country
transactions_by_PSP = transactions.groupby('country')['country'].agg(['count']).sort_values(by='count',ascending=False).reset_index()

sns.countplot(data=transactions, x="country", hue="success")

success_rate_by_PSP =  transactions.pivot_table(values="success",index="country").reset_index()
success_rate_by_PSP['percent'] = (success_rate_by_PSP['success'] * 100).round(2)
success_rate_by_PSP = success_rate_by_PSP.sort_values(by='percent',ascending=False)

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(transactions_by_PSP['count'],labels=transactions_by_PSP['country'], autopct='%.2f%%')
plt.tight_layout()

summary = pd.merge(transactions_by_PSP, success_rate_by_PSP, on="country", how="inner")
display(summary)


#2.4.1.2 Transactions and success rate by PSP
transactions_by_PSP = transactions.groupby('PSP')['PSP'].agg(['count']).sort_values(by='count',ascending=False).reset_index()

success_rate_by_PSP =  transactions.pivot_table(values="success",index="PSP").reset_index()
success_rate_by_PSP['percent'] = (success_rate_by_PSP['success'] * 100).round(2)
success_rate_by_PSP = success_rate_by_PSP.sort_values(by='percent',ascending=False)

summary = pd.merge(transactions_by_PSP, success_rate_by_PSP, on="PSP", how="inner")
display(summary)
display(summary[['count','percent']].corr())

sns.countplot(data=transactions, x="PSP", hue="success")

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(transactions_by_PSP['count'],labels=transactions_by_PSP['PSP'], autopct='%.2f%%')
plt.tight_layout()

#2.4.1.3 Transactions and success rate by 3D seccured
transactions_by_3D_secured = transactions.groupby('3D_secured')['3D_secured'].agg(['count']).sort_values(by='count',ascending=False).reset_index()

success_rate_by_3D_secured =  transactions.pivot_table(values="success",index="3D_secured").reset_index()
success_rate_by_3D_secured['percent'] = (success_rate_by_3D_secured['success'] * 100).round(2)
success_rate_by_3D_secured = success_rate_by_3D_secured.sort_values(by='percent',ascending=False)

summary = pd.merge(transactions_by_3D_secured, success_rate_by_3D_secured, on="3D_secured", how="inner")
display(summary)

sns.countplot(data=transactions, x="3D_secured", hue="success")

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(transactions_by_3D_secured['count'],labels=transactions_by_3D_secured.index, autopct='%.2f%%')
#ax.set_title('Transactions by 3D_secured')
plt.tight_layout()

#2.4.1.4 Transactions and success rate by card

transactions_by_card = transactions.groupby('card')['card'].agg(['count']).sort_values(by='count',ascending=False).reset_index()

success_rate_by_card =  transactions.pivot_table(values="success",index="card").reset_index()
success_rate_by_card['percent'] = (success_rate_by_card['success'] * 100).round(2)
success_rate_by_card = success_rate_by_card.sort_values(by='percent',ascending=False)

summary = pd.merge(transactions_by_card, success_rate_by_card, on="card", how="inner")
display(summary)

sns.countplot(data=transactions, x="card", hue="success")

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(transactions_by_card['count'],labels=transactions_by_card['card'], autopct='%.2f%%')
#ax.set_title('Transactions by card')
plt.tight_layout()

#2.4.2 Numerical Features 

import warnings
warnings.filterwarnings("ignore")

sns.distplot(transactions['amount'], color='blue')
plt.title(f'Distribution of amount')

stats = transactions['amount'].describe()
stats.loc['var'] = transactions['amount'].var()
stats.loc['skew'] = transactions['amount'].skew()
stats.loc['kurt'] = transactions['amount'].kurtosis()
display(stats)

from scipy import stats
p_value = stats.shapiro(transactions['amount'])[1]

#Shapiro-Wilk Test in Python
if p_value <= 0.05:
    print("Null hypothesis of normality is rejected.")
else:
    print("Null hypothesis of normality is accepted.")

#2.4.3 Datetime Feature

transactions['tmsp'] = pd.to_datetime(transactions['tmsp'])
transactions['date'] = transactions['tmsp'].dt.date
transactions['weekday'] = transactions['tmsp'].dt.weekday
transactions['hour'] = transactions['tmsp'].dt.hour

transactions_0 = transactions[transactions.success==0]
df_crosstab = pd.crosstab(transactions_0['tmsp'].dt.weekday, transactions_0['tmsp'].dt.hour)
plt.figure(figsize = (16,5))
sns.heatmap(df_crosstab, annot=True, fmt="d",cmap="YlGnBu", cbar=False, linewidths=.5)

transactions_1 = transactions[transactions.success==1]
df_crosstab = pd.crosstab(transactions_1['tmsp'].dt.weekday, transactions_1['tmsp'].dt.hour)
plt.figure(figsize = (16,5))
sns.heatmap(df_crosstab, annot=True, fmt="d",cmap="YlGnBu", cbar=False, linewidths=.5)

transactions_daily = pd.crosstab(transactions['date'], transactions['success']).reset_index()
transactions_daily.rename(columns={0: "failed", 1: "success"}, inplace = True)
transactions_daily['total'] = transactions_daily['failed'] + transactions_daily['success']
transactions_daily['success_rate'] = (transactions_daily['success'] / transactions_daily['total'] * 100).round(2)
transactions_daily['failure_rate'] = (transactions_daily['failed'] / transactions_daily['total'] * 100).round(2)

transactions_daily.set_index('date')
display(transactions_daily.head())

plt.figure(figsize = (16,5))
sns.lineplot(data=transactions_daily,x="date",y="success_rate")
ax.set(xticks=transactions_daily.date.values)
plt.show()

transactions_daily = pd.crosstab(transactions['weekday'], transactions['success']).reset_index()
transactions_daily.rename(columns={0: "failed", 1: "success"}, inplace = True)
transactions_daily['total'] = transactions_daily['failed'] + transactions_daily['success']
transactions_daily['success_rate'] = (transactions_daily['success'] / transactions_daily['total'] * 100).round(2)
transactions_daily['failure_rate'] = (transactions_daily['failed'] / transactions_daily['total'] * 100).round(2)

transactions_daily.set_index('weekday')
display(transactions_daily)

sns.barplot(data=transactions_daily,x="weekday",y="success_rate")
plt.show()   

#2.4.4 Target variable
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=transactions, x="success")

fig, ax = plt.subplots(figsize=(6, 6))
rates = transactions.groupby('success')['success'].agg(['count'])
ax.pie(rates['count'], labels=['failed','success'], autopct='%.2f%%')
ax.set_title('Success vs Failed Rate')
plt.tight_layout()