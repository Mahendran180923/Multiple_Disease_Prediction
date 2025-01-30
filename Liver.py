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
liver_data = pd.read_csv("indian_liver_patient - indian_liver_patient.csv")
liver_data = pd.DataFrame(liver_data)


liver_data['Albumin_and_Globulin_Ratio'].fillna(liver_data['Albumin_and_Globulin_Ratio'].mean(), inplace=True)


# liver_data.info()

# plt.hist(liver_data['Albumin_and_Globulin_Ratio'], bins=100)
# plt.show()


# sns.boxplot(liver_data['Albumin_and_Globulin_Ratio'])
# plt.title('Box Plot')
# plt.show()



# liver_data.drop(['Gender'], axis=1, inplace=True)
# print(liver_data)


# Find the relationship between the columns 
# plt.figure(figsize=(15,12))
# sns.heatmap(liver_data.corr(),annot=True,vmin=-1, vmax=1)
# plt.show()


# To split the dataset for create a model
x = liver_data.drop(['Gender', 'Dataset'], axis=1)
y = liver_data['Dataset']


# Calculate IQR for each feature
# for feature in x.columns:
#     Q1 = x[feature].quantile(0.25)
#     Q3 = x[feature].quantile(0.75)
#     IQR = Q3 - Q1
#     outliers = x[feature][(x[feature] < Q1 - 1.5 * IQR) | (x[feature] > Q3 + 1.5 * IQR)]
# print(f'Outliers in {feature}: {outliers}')



# Split the data into training and testing sets
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.20, random_state=5)



# to fit in Randon Forest Classifer
rfc_model = RandomForestClassifier(n_estimators=10)
rfc_model.fit(x_train, y_train)
rfc_pred = rfc_model.predict(x_test)
print(f"rfc_score: {rfc_model.score(x_test, y_test)* 100:.2f}%")



# Save the Random Forest Model in Pickle File
pickle.dump(rfc_model, open('rfc_liver_model.pkl','wb'))


# Create a Streamlit app
st.title('Liver Disease Prediction')


# Create input fields for user input
Age = st.slider('Age')
Total_Bilirubin = st.number_input('Total_Bilirubin')
Direct_Bilirubin = st.number_input('Direct_Bilirubin')
Alkaline_Phosphotase = st.number_input('Alkaline_Phosphotase')
Alamine_Aminotransferase = st.number_input('Alamine_Aminotransferase')
Aspartate_Aminotransferase = st.number_input('Aspartate_Aminotransferase')
Total_Protiens = st.number_input('Total_Protiens')
Albumin = st.number_input('Albumin')
Albumin_and_Globulin_Ratio = st.number_input('Albumin_and_Globulin_Ratio')



# Create a button to trigger the prediction
if st.button('Predict'):
#  Load the saved model
    rfc_model = pickle.load(open('rfc_liver_model.pkl', 'rb'))

     # Create a pandas dataframe from user input
    input_data = pd.DataFrame({
        'Age':[Age],
        'Total_Bilirubin': [Total_Bilirubin],
        'Direct_Bilirubin':	[Direct_Bilirubin],
        'Alkaline_Phosphotase':	[Alkaline_Phosphotase],
        'Alamine_Aminotransferase':	[Alamine_Aminotransferase],
        'Aspartate_Aminotransferase': [Aspartate_Aminotransferase],
        'Total_Protiens': [Total_Protiens],
        'Albumin':[Albumin],
        'Albumin_and_Globulin_Ratio':[Albumin_and_Globulin_Ratio]})

    # Make the prediction
    prediction = rfc_model.predict(input_data)


#     # Display the prediction result
    st.write('Prediction:', prediction)


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