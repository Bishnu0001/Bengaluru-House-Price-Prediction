import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error 
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv("C:\\SOFTWARE_BUG_PREDICTION\\s_b_p\\DATASET\\House_prediction_model\\bengaluru_house_prices.csv")

# printing the first 5 element 
df.head()


def convert_sqft(x):
    try:
        if '-' in str(x):
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)  

df = pd.get_dummies(df, drop_first=True) 


# Drop again if new NaN created
df = df.dropna()

# checking the null value 
df.isnull().sum()


df.dropna()

print(df.isnull().sum())

print(df.shape) 

# imbalance dataset

df['price'].value_counts()

print(df.info()) 

# correlation with bugs
corr=df.corr()['price'].sort_values(ascending=False)
print(df.corr()['price'].sort_values(ascending=False))


selected_features = corr[corr >= 0.5]
print(selected_features) 

selected_columns = selected_features.index
print(selected_columns) 

x = df[selected_columns].drop('price', axis=1)
y=df['price']


# Train and test split 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)





model = LinearRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)



rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Accuracy (R2 Score):", r2)
print("RMSE:", rmse)
print(y.mean()) 












