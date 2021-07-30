import streamlit as st
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



st.title("Data Analytics Project by De Winn")
'To Predict If a User Would Make a Hotel Bookings based on the Several Factors.'
option = st.sidebar.selectbox(
    'Select a Model',
     ['Decision Tree','KNN','SVM','The End'])

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
                
                img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

                image = Image.open(img_file_buffer)
                img_array = np.array(image)

                if image is not None:
                    st.image(
                       image,
                       caption=f"You amazing image has shape {img_array.shape[0:2]}",
                       use_column_width=True,)
                
                '4) Resampling through DownSampling using NearMiss'
                '- Change X & y into np array'
                '- DownSampling the 0 in y'
                '- Concatenate by Stacking the 2 Arrays'
                '- Transform array into pandas dataframe'
                st.set_option('deprecation.showPyplotGlobalUse', False)
                sns.countplot(df['is_booking'])
                st.pyplot()
                

                '5) Train Test Split'
                X = df.drop('is_booking', axis=1)
                y = df['is_booking']

                Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=1)

                sc = StandardScaler()
                X_train = sc.fit_transform(Xtrain)
                X_test = sc.transform(Xtest)

                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier().fit(X_train, ytrain)

                ypred = clf.predict(X_test)

                from sklearn.metrics import classification_report, confusion_matrix
                print(confusion_matrix(ytest, ypred))
                print()
                print()
                print(classification_report(ytest, ypred))
                st.write(classification_report(ytest, ypred))

                '6) Feature Importances'
                clf.feature_importances_

                important_factors = pd.DataFrame({'Factor': list(X.columns), 'Importance': clf.feature_importances_})
                important_factors.sort_values(by=['Importance'], ascending=False,inplace=True)
                important_factors

                'Conclusion:' 
                '- searched destination,'
                '- search count & '
                '- the distance between origin & destinations '
                'are the top 3 factors that determine booking decision.'







    elif option=='KNN':

        st.header("KNN")
        'Please upload your data in .csv format here'
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
                '1) Let\'s take a look at the first 5 rows of the data.'
                df = pd.read_csv(uploaded_file)
                st.write(df.head())

        X = df.drop('is_booking', axis=1)
        y = df['is_booking']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=1)

        sc = StandardScaler()
        X_train = sc.fit_transform(Xtrain)
        X_test = sc.transform(Xtest)

        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier()
        knn.fit(X_train, ytrain)
        ypred = knn.predict(X_test)

        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(ytest, ypred))
        print()
        print()
        print(classification_report(ytest, ypred))
        st.write(classification_report(ytest, ypred))

    elif option=='SVM':

        st.header("SVM")
        'Please upload your data in .csv format here'
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
                '1) Let\'s take a look at the first 5 rows of the data.'
                df = pd.read_csv(uploaded_file)
                st.write(df.head())

        X = df.drop('is_booking', axis=1)
        y = df['is_booking']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=1)

        sc = StandardScaler()
        X_train = sc.fit_transform(Xtrain)
        X_test = sc.transform(Xtest)
    
        from sklearn.svm import SVC
        svm = SVC()
        svm.fit(X_train, ytrain)
        ypred = svm.predict(X_test)

        print(confusion_matrix(ytest, ypred))
        print()
        print()
        print(classification_report(ytest, ypred))
        st.write(classification_report(ytest, ypred))


    else:
        'Starting a long computation...'

    
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
   
            latest_iteration.text(f'Iteration {i+1}')
            bar.progress(i + 2)
            time.sleep(0.05)

            '...That\'s The End, Thank You!'
            
