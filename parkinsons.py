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
import pickle
import streamlit as st


# Load Parkinsons Dataset
parkinsons_data = pd.read_csv("parkinsons - parkinsons.csv")
parkinsons_data = pd.DataFrame(parkinsons_data)
# print(parkinsons_data)
# parkinsons_data.info()

x = parkinsons_data.drop(['name', 'status'], axis=1)
y = parkinsons_data['status']
# print(parkinsons_feature)
# print(parkinsons_target)


# Calculate IQR for each feature
# for feature in x.columns:
#     if feature != 'status':  # exclude the target variable
#         Q1 = x[feature].quantile(0.25)
#         Q3 = x[feature].quantile(0.75)
#         IQR = Q3 - Q1
#         outliers = x[feature][(x[feature] < Q1 - 1.5 * IQR) | (x[feature] > Q3 + 1.5 * IQR)]
#         print(f'Outliers in {feature}: {outliers}')


# Split the data into training and testing sets
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.20, random_state=3)


# to fit in Randon Forest Classifer
rfc_model = RandomForestClassifier(n_estimators=10)
rfc_model.fit(x_train, y_train)
rfc_pred = rfc_model.predict(x_test)
print(f"rfc_score: {rfc_model.score(x_test, y_test)* 100:.2f}%")


# Save the Random Forest Model in Pickle File
pickle.dump(rfc_model, open('rfc_parkinsons_model.pkl','wb'))



# # Create a Streamlit app
st.title('Parkinsons Disease Prediction')


#  Create input fields for user input
MDVP_Fo = st.number_input('MDVP:Fo(Hz)')
MDVP_Fhi = st.number_input('MDVP:Fhi(Hz)')
MDVP_Flo = st.number_input('MDVP:Flo(Hz)')
MDVP_Jitter_percentage = st.number_input('MDVP:Jitter(%)')
MDVP_Jitter_Abs = st.number_input('MDVP:Jitter(Abs)')
MDVP_Rap = st.number_input('MDVP:RAP')
MDVP_Ppq = st.number_input('MDVP:PPQ')
Jitter_DDP = st.number_input('Jitter:DDP')
MDVP_Shimmer = st.number_input('MDVP:Shimmer')
MDVP_Shimmer_dB = st.number_input('MDVP:Shimmer(dB)')
Shimmer_APQ3 = st.number_input('Shimmer:APQ3')
Shimmer_APQ5 = st.number_input('Shimmer:APQ5')
MDVP_Apq= st.number_input('MDVP:APQ')
Shimmer_DDA = st.number_input('Shimmer:DDA')
NHR = st.number_input('NHR')
HNR = st.number_input('HNR')
RPDE = st.number_input('RPDE')
DFA = st.number_input('DFA')
spread1 = st.number_input('spread1')
spread2 = st.number_input('spread2')
D2 = st.number_input('D2')
PPE = st.number_input('PPE')


# # Create a button to trigger the prediction
if st.button('Predict'):
# # Load the saved model
    rfc_model = pickle.load(open('rfc_parkinsons_model.pkl', 'rb'))

#     # Create a pandas dataframe from user input
    input_data = pd.DataFrame({
        'MDVP:Fo(Hz)': [MDVP_Fo], 'MDVP:Fhi(Hz)': [MDVP_Fhi], 'MDVP:Flo(Hz)': [MDVP_Flo], 'MDVP:Jitter(%)': [MDVP_Jitter_percentage],
       'MDVP:Jitter(Abs)': [MDVP_Jitter_Abs], 'MDVP:RAP' :[MDVP_Rap], 'MDVP:PPQ': [MDVP_Ppq], 'Jitter:DDP': [Jitter_DDP] ,
       'MDVP:Shimmer': [MDVP_Shimmer], 'MDVP:Shimmer(dB)': [MDVP_Shimmer_dB], 'Shimmer:APQ3': [Shimmer_APQ3], 'Shimmer:APQ5': [Shimmer_APQ5],
       'MDVP:APQ': [MDVP_Apq], 'Shimmer:DDA': [Shimmer_DDA], 'NHR': [NHR], 'HNR':[HNR], 'RPDE':[RPDE], 'DFA':[DFA],
       'spread1':[spread1], 'spread2':[spread2], 'D2':[D2], 'PPE': [PPE]})

#     # Make the prediction
    prediction = rfc_model.predict(input_data)


#     # Display the prediction result
    st.write('Prediction:', prediction)











# Other models 

# Fit a Linear Regression Model
# linear_regression_model = LinearRegression()
# linear_regression_model.fit(x_train,  y_train)
# liner_regression_pred = linear_regression_model.predict(x_test)


# Fit a Decision Tree
# dtc = DecisionTreeClassifier()
# dtc.fit(x_train, y_train)
# dtc_pred = dtc.predict(x_test)



# various score comparison for linear regression
# linear_regression_mse = mean_squared_error(y_test, liner_regression_pred) * 100
# linear_regression_mae = mean_absolute_error(y_test, liner_regression_pred) * 100
# linear_regression_rmse = root_mean_squared_error(y_test, liner_regression_pred) * 100
# linear_regression_r2 = r2_score(y_test, liner_regression_pred) * 100


# various score comparison for decision tree
# dtc_mse = mean_squared_error(y_test, dtc_pred) * 100
# dtc_mae = mean_absolute_error(y_test, dtc_pred) * 100
# dtc_rmse = root_mean_squared_error(y_test, dtc_pred) * 100
# dtc_r2 = r2_score(y_test, dtc_pred) * 100


# to fit in KNN Classifier 
# knn = KNeighborsClassifier(n_neighbors=3)
# bc = BaggingClassifier(estimator=knn,n_estimators=10,random_state=1)

# bc.fit(x_train,y_train)
# knn_pred = bc.predict(x_test)


# Gradient Boosting
# gbc = GradientBoostingClassifier(learning_rate=0.1)
# gbc.fit(x_train,y_train)
# gbc_pred = gbc.predict(x_test)


# print(f"mse: {linear_regression_mse:.2f}%,\n mae: {linear_regression_mae:.2f}%,\n rmse: {linear_regression_rmse:.2f}%,\n r2: {linear_regression_r2:.2f}%")
# print(f"score: {linear_regression_model.score(x_test, y_test)* 100:.2f}%")

# print(f"mse: {dtc_mse:.2f}%,\n mae: {dtc_mae:.2f}%,\n rmse: {dtc_rmse:.2f}%,\n r2: {dtc_rmse:.2f}%")
# print(f"dtc_score: {dtc.score(x_test, y_test)* 100:.2f}%")


# plot_tree(rfc.estimators_[1],filled=True)  #first tree
# plt.show()
# print(f"knn_score: {accuracy_score(y_test,knn_pred) * 100:.2f}%")

# print(f"gbc_score: {accuracy_score(y_test,gbc_pred) * 100:.2f}%")