import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, root_mean_squared_error
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st


# Load Parkinsons Dataset
kidney_data = pd.read_csv("kidney_disease - kidney_disease.csv")
kidney_data = pd.DataFrame(kidney_data)

print(kidney_data)


# Find the relationship between the columns 
# plt.figure(figsize=(15,12))
# sns.heatmap(liver_data.corr(),annot=True,vmin=-1, vmax=1)
# plt.show()


# To split the dataset for create a model
# x = liver_data.drop(['Gender', 'Dataset'], axis=1)
# y = liver_data['Dataset']


# Calculate IQR for each feature
# for feature in x.columns:
#     Q1 = x[feature].quantile(0.25)
#     Q3 = x[feature].quantile(0.75)
#     IQR = Q3 - Q1
#     outliers = x[feature][(x[feature] < Q1 - 1.5 * IQR) | (x[feature] > Q3 + 1.5 * IQR)]
# print(f'Outliers in {feature}: {outliers}')



# Split the data into training and testing sets
# x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.20, random_state=5)



# to fit in Randon Forest Classifer
# rfc_model = RandomForestClassifier(n_estimators=10)
# rfc_model.fit(x_train, y_train)
# rfc_pred = rfc_model.predict(x_test)
# print(f"rfc_score: {rfc_model.score(x_test, y_test)* 100:.2f}%")



# Save the Random Forest Model in Pickle File
# pickle.dump(rfc_model, open('rfc_kidney_model.pkl','wb'))




# # Fit a Linear Regression Model
# linear_regression_model = LinearRegression()
# linear_regression_model.fit(x_train,  y_train)
# liner_regression_pred = linear_regression_model.predict(x_test)

# # various score comparison for linear regression
# linear_regression_mse = mean_squared_error(y_test, liner_regression_pred) * 100
# linear_regression_mae = mean_absolute_error(y_test, liner_regression_pred) * 100
# linear_regression_rmse = root_mean_squared_error(y_test, liner_regression_pred) * 100
# linear_regression_r2 = r2_score(y_test, liner_regression_pred) * 100


# print(f"mse: {linear_regression_mse:.2f}%,\n mae: {linear_regression_mae:.2f}%,\n rmse: {linear_regression_rmse:.2f}%,\n r2: {linear_regression_r2:.2f}%")
# print(f"score: {linear_regression_model.score(x_test, y_test)* 100:.2f}%")





# Gradient Boosting
# gbc = GradientBoostingClassifier(learning_rate=0.1)
# gbc.fit(x_train,y_train)
# gbc_pred = gbc.predict(x_test)


# print(f"gbc_score: {accuracy_score(y_test,gbc_pred) * 100:.2f}%")


# # to fit in KNN Classifier 
# knn = KNeighborsClassifier(n_neighbors=3)
# bc = BaggingClassifier(estimator=knn,n_estimators=10,random_state=1)

# bc.fit(x_train,y_train)
# knn_pred = bc.predict(x_test)


# print(f"knn_score: {accuracy_score(y_test,knn_pred) * 100:.2f}%")

# Fit a Decision Tree
# dtc = DecisionTreeClassifier()
# dtc.fit(x_train, y_train)
# dtc_pred = dtc.predict(x_test)

# # # various score comparison for decision tree
# dtc_mse = mean_squared_error(y_test, dtc_pred) * 100
# dtc_mae = mean_absolute_error(y_test, dtc_pred) * 100
# dtc_rmse = root_mean_squared_error(y_test, dtc_pred) * 100
# dtc_r2 = r2_score(y_test, dtc_pred) * 100

# print(f"dtc_score: {dtc.score(x_test, y_test)* 100:.2f}%")