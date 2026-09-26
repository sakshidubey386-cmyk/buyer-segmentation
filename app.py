import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("Parcel Buyer Segmentation")

uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Age column fix for date_of_birth
    if 'date_of_birth' in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        df['Age'] = 2025 - df['date_of_birth'].dt.year
    elif 'YOB' in df.columns:
        df['Age'] = 2025 - df['YOB']
    elif 'Year_Birth' in df.columns:
        df['Age'] = 2025 - df['Year_Birth']
    else:
        st.error(f"Age column nahi mila. Columns: {list(df.columns)}")
        st.stop()

    # Simple clustering on Age (add more columns if available)
    df_clean = df[['Age']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("Buyer Segments (4 Clusters)")
    cluster_counts = df_clean['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    st.write("Cluster 0: Young buyers")
    st.write("Cluster 1: Mature buyers") 
    st.write("Cluster 2: Middle-aged buyers")
    st.write("Cluster 3: Senior buyers")
    
   
