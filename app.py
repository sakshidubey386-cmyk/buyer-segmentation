import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Parcel Buyer Segmentation")

uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    if 'date_of_birth' in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        df['Age'] = 2025 - df['date_of_birth'].dt.year
    elif 'YOB' in df.columns:
        df['Age'] = 2025 - df['YOB']
    elif 'Year_Birth' in df.columns:
        df['Age'] = 2025 - df['Year_Birth']
    else:
        st.error("Age column nahi mila")
        st.stop()

    df_clean = df[['Age']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("Buyer Segments")
    st.bar_chart(df_clean['Cluster'].value_counts().sort_index())
    st.success("Ho gaya! 4 clusters ban gaye.")
   
