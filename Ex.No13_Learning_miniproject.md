# Ex.No: 13 Machine Learning - Mini Project  
### DATE: 04/11/24                                                                           
### REGISTER NUMBER : 212221220041
### AIM: 
To write a program to train the classifier for Wine Quality Prediction.
###  Algorithm:
1. Load the dataset and split it into features (`x`) and target (`y`).
2. Split the dataset into training and testing sets using `train_test_split`.
3. Scale the features using `StandardScaler` for both training and testing data.
4. Instantiate the MLP (Multilayer Perceptron) classifier and train it on the scaled training data.
5. Build a Gradio interface to input values for the model, predict diabetes outcomes, and display "Good" or "Bad" based on the prediction.

### Program:
```
import pandas as pd
df=pd.read_csv('car data.csv')
df.shape
print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
df.isnull().sum()
df.describe()
final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
final_dataset.head()
final_dataset['Current Year']=2020
final_dataset.head()
final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']
final_dataset.head()
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.head()
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset.head()
final_dataset=final_dataset.drop(['Current Year'],axis=1)
final_dataset.head()
final_dataset.corr()
import seaborn as sns
sns.pairplot(final_dataset)
import seaborn as sns
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
X['Owner'].unique()
X.head()
y.head()
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions=rf_random.predict(X_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

<br> <br> <br> <br> <br> <br> <br> <br>

### Output:
<img src="https://github.com/user-attachments/assets/c885b031-38f5-4861-9a35-08f1c8e43464" height=400>
<img src="https://github.com/user-attachments/assets/4cc8a4f8-7071-4dbb-acc7-db1f22af06d8" height=400>

<br> <br>

### Result:
Thus the system was trained successfully and the prediction was carried out.
