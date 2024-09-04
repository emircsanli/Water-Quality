import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('water_potability.csv')

print(data.sample(2))

print(data.shape)

print(data.isna().sum())

sns.heatmap(data.corr().abs(),cmap='viridis')
plt.show()

sns.pairplot(data)
plt.show()

show1=data['Potability'].value_counts().plot.bar(color='brown')

print(show1)

print(data['ph'].mean())
data['ph']=data['ph'].fillna(data['ph'].mean())

print(data.ph.describe())
print(data[data.ph>9].shape)
print(data[data.ph<6].shape)

data['ph']=data['ph'].apply(lambda x : 9 if x>9 else x)
data['ph']=data['ph'].apply(lambda x : 6 if x<6 else x)
sns.violinplot(data=data, x='Potability', y='ph', palette=['brown','yellow'])
plt.show()
print(data[data.ph<7].groupby('Potability').Potability.value_counts())
print(data[data.ph>7].groupby('Potability').Potability.value_counts())

plt.subplot(1, 2, 1)
data[data['ph'] < 7]['Potability'].value_counts().plot.pie(
    autopct='%1.1f%%',
    colors=['#ecf5a3', '#efc458'] )
plt.title('pH < 7')
plt.ylabel('')

plt.subplot(1, 2, 2)
data[data['ph'] > 7]['Potability'].value_counts().plot.pie(
    autopct='%1.1f%%',
    colors=['#ecf5a3', '#efc458']
)
plt.title('pH > 7')
plt.tight_layout()
plt.ylabel('')

plt.show()

print(data.groupby('Potability').ph.describe())
sns.barplot(data,x='Potability',y='Hardness')
plt.show()
sns.boxplot(data,x='Potability',y='Hardness')
plt.show()
print(data.Hardness.describe())

print(data[data['Hardness']>230].shape)
data['Hardness']=data['Hardness'].apply(lambda x : 230 if x >230 else x )
print(data.Hardness)
print(data[data['Hardness']<160].shape)
data['Hardness']=data['Hardness'].apply(lambda x : 160 if x <160 else x)
print(data.Hardness)
sns.boxplot(data,x='Potability',y='Hardness')
plt.show()
print(data.groupby('Potability').Solids.mean())
sns.barplot(data,x='Potability',y='Solids')
plt.show()
sns.violinplot(data,x='Potability',y='Solids')
plt.show()
print(data.Chloramines.describe())
print(data.groupby('Potability').Chloramines.describe())
print(data.groupby('Potability').Chloramines.mean().plot.bar())
data=data.drop(columns='Chloramines')
print(data['Sulfate'].describe())

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
sns.barplot(data,x='Potability',y='Sulfate',palette=['#ecf5a3', '#efc458'])
plt.subplot(1,3,2)
sns.violinplot(data,x='Potability',y='Sulfate',palette=['#ecf5a3', '#efc458'])
plt.subplot(1,3,3)
sns.boxplot(data,x='Potability',y='Sulfate',palette=['#ecf5a3', '#efc458'])

data['Sulfate']=data['Sulfate'].fillna(data['Sulfate'].mean())
print(data['Conductivity'].describe())

print(data.groupby('Potability').Conductivity.describe())
print(data[data['Conductivity']>500].shape)
data['Conductivity']=data['Conductivity'].apply(lambda x : 500 if x>500 else x)
print(data[data['Conductivity']<300].shape)
data['Conductivity']=data['Conductivity'].apply(lambda x : 300 if x <300 else x)
print(data.groupby('Potability').Conductivity.median())


plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
sns.barplot(data,x='Potability',y='Organic_carbon',palette=['#ecf5a3', '#efc458'])
plt.subplot(1,3,2)
sns.violinplot(data,x='Potability',y='Organic_carbon',palette=['#ecf5a3', '#efc458'])
plt.subplot(1,3,3)
sns.boxplot(data,x='Potability',y='Organic_carbon',palette=['#ecf5a3', '#efc458'])

data=data.drop(columns='Organic_carbon')
print(data.Trihalomethanes.describe())

data['Trihalomethanes']=data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())
print(data.groupby('Potability').Trihalomethanes.describe())

data=data.drop(columns='Trihalomethanes')
print(data.groupby('Potability')['Turbidity'].describe())

print(data.Turbidity.describe())
data.sample(2)

print(data.groupby('Potability').Potability.value_counts())
print(data.shape)

Y=data['Potability']
X=data.drop(columns='Potability')
scaler=RobustScaler()
pca=PCA(.95)
x_scal=scaler.fit_transform(X)
X_scal=scaler.fit_transform(x_scal)
X=pd.DataFrame(X_scal,columns=X.columns)
print(X.sample(2))
Y=pd.DataFrame(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=23)
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
accuracy_score(Y_test,y_pred)
print(classification_report(Y_test,y_pred))
sns.heatmap(confusion_matrix(Y_test,y_pred),annot=True)
plt.show()
print("model's decision tree point",model.score(X_test,Y_test))

model=RandomForestClassifier()
model.fit(X_train,Y_train)
print("model's random forest point",model.score(X_test,Y_test))

model=SVC()
model.fit(X_train,Y_train)
print("model's SVM point",model.score(X_test,Y_test))

model=KNeighborsClassifier(4)
model.fit(X_train,Y_train)
print("model's KNN point",model.score(X_test,Y_test))

cv_score=cross_val_score(model,X,Y,cv=5)
print("cross val's score",cv_score)
print("cross vall mean score",cv_score.mean())

model=XGBClassifier()
model.fit(X_train,Y_train)
print("model's XGB point",model.score(X_test,Y_test))



"""""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('water_potability.csv')

# Basic data inspection
print(data.sample(2))
print(data.shape)
print(data.isna().sum())

# Data visualization
sns.heatmap(data.corr().abs(), cmap='viridis')
plt.show()

sns.pairplot(data)
plt.show()

data['Potability'].value_counts().plot.bar(color='brown')
plt.show()

# Data cleaning and preprocessing
data['ph'] = data['ph'].fillna(data['ph'].mean())
data['ph'] = data['ph'].apply(lambda x: 9 if x > 9 else (6 if x < 6 else x))
sns.violinplot(data=data, x='Potability', y='ph', palette=['brown','yellow'])
plt.show()

data['Hardness'] = data['Hardness'].apply(lambda x: 230 if x > 230 else (160 if x < 160 else x))
sns.boxplot(data=data, x='Potability', y='Hardness')
plt.show()

data = data.drop(columns='Chloramines')
data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
data['Conductivity'] = data['Conductivity'].apply(lambda x: 500 if x > 500 else (300 if x < 300 else x))

data = data.drop(columns=['Organic_carbon', 'Trihalomethanes'])

# Splitting the data into features and target
Y = data['Potability']
X = data.drop(columns='Potability')

# Applying scaling and PCA
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)
X = pd.DataFrame(X_pca)

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=23)

# Define a function to evaluate models
def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy_score(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True)
    plt.show()

# Decision Tree
model = DecisionTreeClassifier()
evaluate_model(model, X_train, Y_train, X_test, Y_test)

# Random Forest
model = RandomForestClassifier()
evaluate_model(model, X_train, Y_train, X_test, Y_test)

# SVM
model = SVC()
evaluate_model(model, X_train, Y_train, X_test, Y_test)

# KNN
model = KNeighborsClassifier(4)
evaluate_model(model, X_train, Y_train, X_test, Y_test)

# Cross-validation for KNN
cv_score = cross_val_score(model, X, Y, cv=5)
print("Cross val scores:", cv_score)
print("Cross val mean score:", cv_score.mean())

# XGBoost
model = XGBClassifier()
evaluate_model(model, X_train, Y_train, X_test, Y_test)

# Hyperparameter Tuning for Random Forest (example)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)
print("Best parameters for Random Forest:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_
evaluate_model(best_rf_model, X_train, Y_train, X_test, Y_test)
"""""