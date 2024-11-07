import streamlit as st
import pandas as pd
import pickle
import os
import plotlt.express as px
import numpy as np
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go

def scatter(model, model_name, data, new_point, features, color_scale, title):
    clusters = model.fit_predict(data[features])
    data[f"{model_name}_Cluster"] = clusters

    if model_name == "KMeans_model":
        new_cluster = model.predict(new_point[features])[0]
    else:
        distances = pairwise_distances(new_point[features])[0]
        nearest_index = distances.argmin()
        new_cluster = clusters[nearest_index]

    fig = px.scatter_3d(data, x='Avf_Credit_Limit', y='Total_Credit_Cards', z='Total_visits_online',
                        color=f"{model_name}_Cluster", title=title, color_continous_scale=color_scale)
    
    fig.add_trace(
        go.scatter3d(
            x=new_point['Avg_Credit_Limit'],
            y=new_point['Total_Credit_Cards'],
            z=new_point['Total_visits_online'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='New Point',
        )
    )
    return fig, new_cluster

st.set_page_config(
    page_title="11805 - Unsupervised Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.markdown("<h1 style='text-align: center;'>Unsupervised Learning - 11805</h1>", unsafe_allow_html=True)
    st.dataframe(input_data)

    model_directory = r'C:\Users\ASUS\Documents\2024\ML\Tugas4_B_11805'
    model_path = {
        "AGG_model": os.path.join(model_directory, r'AGG_model.pkl'),
        "KMeans_model": os.path.join(model_directory, r'KMeans_model.pkl'),
        "DBSCAN_model": os.path.join(model_directory, r'DBSCAN_model.pkl'),
    }

    models = {}
    for model_name, path in model_path.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
        else:
            st.write(f"Model {model_name} tidak ditemukan di path : ", path)

    avg_cl = st.sidebar.number_input("Average Credit Limit", 0, 10000)
    sum_cc = st.sidebar.number_input("Average Credit Cards", 0, 10)
    sum_vo = st.sidebar.number_input("Total Visits Online", 0, 16)

    if st.sidebar.button("Prediksi !"):
        features = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_Online']
        new_point = pd.DataFrame([[avg_cl, sum_cc, sum_vo]], columns=features)
        cluster_method = [
            ("KMeans_model", "models['KMeans_model']", "KMeans Clustering", px.colors.sequential.Cividis),
            ("AGG_model", "models['AGG_model']", "Agglomerative Clustering", px.colors.sequential.Mint),
            ("DBSCAN_model", "models['DBSCAN_model']", "DBSCAN Clustering", px.colors.sequential.Plasma)
        ]

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, (model_name, model, title, color_scale) in enumerate(cluster_method):
            fig, new_cluster = scatter(model, model_name, input_data, new_point, features, color_scale, title)
            with cols[i]:
                st.plotly_chart(fig)
                st.markdown(f"<p style='text-align: center;'>Titik Data Baru yang baru masuk ke dalam cluster: {new_cluster}</p>", unsafe_allow_html=True)

