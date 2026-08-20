import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
st.title("👥 Customer Segmentation Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Online_Retail_Customer_Segmentation.csv')
    return df

df = load_data()

# Sidebar
st.sidebar.header("📊 Menu")
page = st.sidebar.selectbox("Choose Page:", ["Home", "Data Overview", "Segments", "Predict"])

# Load model
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Home Page
if page == "Home":
    st.header("🎯 Welcome to Customer Segmentation System")
    st.markdown(f"""
    ### About This Project
    This application uses machine learning to segment customers based on their 
    purchasing behavior using **RFM Analysis**:
    
    - **Recency**: How recently a customer made a purchase
    - **Frequency**: How often they make purchases
    - **Monetary**: How much money they spend
    
    ### Dataset Information
    - Total Customers: {len(df)}
    - Countries: {df['Country'].nunique()}
    """)

# Data Overview
elif page == "Data Overview":
    st.header("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Countries", df['Country'].nunique())
    with col3:
        st.metric("Avg Spending", f"£{df['Monetary_TotalSpend_GBP'].mean():.2f}")
    st.dataframe(df.head(10))

# Segments
elif page == "Segments":
    st.header("👥 Customer Segments")
    features = ['Recency_Days', 'Frequency_Transactions', 'Monetary_TotalSpend_GBP']
    X = df[features]
    X_scaled = scaler.transform(X)
    df['Cluster'] = model.predict(X_scaled)
    
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("### Segment Distribution")
    st.write(cluster_counts)
    
    st.write("### Segment Profiles")
    segment_profiles = df.groupby('Cluster')[features].mean()
    st.dataframe(segment_profiles)

# Predict
elif page == "Predict":
    st.header("🔮 Predict Customer Segment")
    col1, col2 = st.columns(2)
    with col1:
        recency = st.number_input("Recency (Days)", min_value=0, max_value=365, value=30)
        frequency = st.number_input("Frequency", min_value=1, max_value=100, value=5)
    with col2:
        monetary = st.number_input("Monetary (GBP)", min_value=0.0, value=500.0)
    
    if st.button("Predict Segment", type="primary"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"**Predicted Segment: {prediction}**")