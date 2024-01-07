import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

def handle_outliers(column, df):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, np.where(df[column] > upper_bound, upper_bound, df[column]))

df = pd.read_csv('pendapatan.csv')

df.dropna(inplace=True)

df = handle_outliers('jumlahh_pendapatan', df)

selected_columns = ['jumlah_pendapatan', 'tahun']
X = df[selected_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.sidebar.title("K-MEANS CLUSTERING")
st.sidebar.markdown("NOMOR CLUSTERS:")
num_clsuters = st.sidebar.text_input("NOMOR CLUSTERS", "3")

num_clusters = int(num_clusters)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

selected_visualization = st.sidebar.selectbox("Select Visualization", ["Clustered Data", "Trend Analysis", "Top Income by Year", "Top Income by Region"])

st.title("K-MEANS CLUSTERIN JUMLAH PENDAPATAN PARIWISATA")

st.subheader("DATA")
st.write(df)

if selected_visualization == "Clustered Data":
    st.subheader("Visualisasi Clustering")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.scatterplot(x='tahun', y='jumlah_pendapatan', hue='cluster', data=df, palette='viridis', ax=ax)
