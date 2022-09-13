import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

df_boston = pd.read_csv('Boston House Prices.csv')
df_boston


y = df_boston['Value'] # dependent variable
x = df_boston['Rooms'] # independent variable

x = sm.add_constant(x) # adding a constant
lm = sm.OLS(y,x).fit() # fitting the model

lm.predict(x)

print(lm.predict(x))
print(lm.summary())

# Rooms coef: 9.1021
# Constant coef: - 34.6706
y_pred = 9.1021 * x['Rooms'] - 34.6706

# plotting the data points
sns.scatterplot(x=x['Rooms'], y=y)

#plotting the line
sns.lineplot(x=x['Rooms'],y=y_pred, color='red')

#axes
plt.xlim(0)
plt.ylim(0)
plt.show()