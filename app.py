import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

st.title("Parcel Buyer Segmentation")
st.write("Upload your parcel_data.csv to see buyer clusters")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    df['Age'] = 2025 - df['YOB']
    df['Tenure'] = 2025 - df['Customer_Since']
    X = df[['Age', 'Tenure', 'Product_Category', 'Gender', 'Annual_Spend']]
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(X[['Product_Category', 'Gender']])
    import numpy as np
    X_num = X[['Age', 'Tenure', 'Annual_Spend']].values
    X_final = np.hstack([X_num, encoded])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    st.subheader("Clustered Data")
    st.dataframe(df.head(10))
    
    st.subheader("Cluster Counts")
    st.bar_chart(df['Cluster'].value_counts())
    st.success("Segmentation Done! 4 Buyer Segments Found.")
else:
    st.info("Please upload parcel_data.csv file")
