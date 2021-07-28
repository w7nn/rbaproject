import streamlit as st

import numpy as np
import pandas as pd

st.header("My first Streamlit App")
st.write(pd.DataFrame({
    'Intplan': ['yes', 'yes', 'yes', 'no'],
    'Churn Status': [0, 0, 0, 1]
}))

[[18203   222]
 [ 1703    41]]


              precision    recall  f1-score   support

           0       0.91      0.99      0.95     18425
           1       0.16      0.02      0.04      1744

    accuracy                           0.90     20169
   macro avg       0.54      0.51      0.50     20169
weighted avg       0.85      0.90      0.87     20169