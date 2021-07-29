import streamlit as st
import numpy as np
import pandas as pd
import time



st.title("RBA Project")
'To Predict If a User Would Do a Hotel Bookings based on several factors.'
option = st.sidebar.selectbox(
    'Select a Model',
     ['Decision Tree','KNN','Gaussian Naive Bayes','SVM'])

st.write('Before you continue, please read the [terms and conditions](https://www.gnu.org/licenses/gpl-3.0.en.html)')
show = st.checkbox('I agree the terms and conditions')
if show:

    if option=='Decision Tree':

        st.header("Decision Tree")
        'Please upload your data in .csv format here'
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
                '1) Let\'s take a look at the first 5 rows of the data.'
                df = pd.read_csv(uploaded_file)
                st.write(df.head())
                
                '2) Ensure there are no Null'
                st.write(df.isnull().sum())
                
                '3) Let\'s take a look at our Label(y)'
                '0 = No Booking; 1 = Booking'
                df['is_booking'].value_counts()
                #sns.countplot(df['is_booking'])
                plt.show(sns.countplot(df['is_booking']))

                '4) Too see if there is any Gaussian distribution'
                import pylab as pl
                df.drop('is_booking' ,axis=1).hist(bins=100, figsize=(10,10))
                pl.suptitle("Histogram for each numeric input variable")
                plt.savefig('hotel2_hist')
                plt.show()

                '5) Train Test Split'
                X_hotel2 = df.drop('is_booking', axis=1)
                y_hotel2 = df['is_booking']

                Xtrain, Xtest, ytrain, ytest = train_test_split(X_hotel2,y_hotel2,test_size=0.2,random_state=1)
                Xtrain, Xtest, ytrain, ytest = train_test_split(X_hotel2,y_hotel2,test_size=0.2,random_state=1)
                sc = StandardScaler()
                X_train = sc.fit_transform(Xtrain)
                X_test = sc.transform(Xtest)

                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier().fit(X_train, y_train)

                #clf = DecisionTreeClassifier()
                #clf.fit(Xtrain, ytrain)
                y_pred = clf.predict(X_test)

                from sklearn.metrics import classification_report, confusion_matrix
                print(confusion_matrix(y_test, y_pred))
                print()
                print()
                print(classification_report(y_test, y_pred))

                clf.feature_importances_

                important_factors = pd.DataFrame({'Factor': list(X.columns), 'Importance': clf.feature_importances_})
                important_factors.sort_values(by=['Importance'], ascending=False,inplace=True)
                important_factors



    elif option=='KNN':
        ap_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])

        st.map(map_data)

    elif option=='T n C':

    
        st.write(pd.DataFrame({
        'Intplan': ['yes', 'yes', 'yes', 'no'],
        'Churn Status': [0, 0, 0, 1]
         }))


    else:
        'Starting a long computation...'

    
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
   
            latest_iteration.text(f'Iteration {i+1}')
            bar.progress(i + 1)
            time.sleep(0.05)

            '...and now we\'re done!'
