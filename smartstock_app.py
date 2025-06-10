import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pandas.tseries.offsets import DateOffset
import plotly.express as px  # For interactive visualization

# Page configuration
st.set_page_config(
    page_title="Akshitha’s SmartStock Forecasting App",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for UI enhancements
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    .header-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        width: 100%;
        background-color: #000000;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .tagline {
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.5rem;
        text-align: center;
        background: linear-gradient(90deg, #4299e1 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: #ffffff;
    }

    label {
        color: #ffffff !important;
    }

    .stCard {
        background-color: #1a202c;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton > button {
        background: linear-gradient(90deg, #4299e1 0%, #667eea 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
    }

    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #000000;
        color: #4299e1;
        text-align: center;
        padding: 15px;
        border-top: 1px solid #2d3748;
        z-index: 1000;
    }

    .footer-icons a {
        margin: 0 10px;
        text-decoration: none;
        color: #ffffff;
        transition: color 0.3s ease;
    }

    .footer-icons a:hover {
        color: #007acc;
    }

    .footer-icons img {
        width: 20px;
        height: 20px;
        vertical-align: middle;
    }

    @media screen and (max-width: 768px) {
        .stApp {
            padding-top: 150px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
# Header Section - Customized
# Header Section - Customized
st.markdown('<div class="header-container">', unsafe_allow_html=True)

import os
if os.path.exists("newlogo.png"):
    st.image("newlogo.png", caption="Akshitha Analytics", width=1100)
else:
    st.warning("Logo image not found. Please make sure 'newlogo.png' is in the app directory.")

st.markdown('<h1 class="tagline">Intelligent Forecasting & Analysis</h1>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# Main Content Container
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<h3>Data Input</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your time series data (CSV)", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown('<h3>Model Parameters</h3>', unsafe_allow_html=True)
        p = st.number_input("AR(p) Order", min_value=0, max_value=5, value=1)
        d = st.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("MA(q) Order", min_value=0, max_value=5, value=1)
        seasonal_p = st.number_input("Seasonal AR Order (P)", min_value=0, max_value=5, value=1)
        seasonal_d = st.number_input("Seasonal Differencing (D)", min_value=0, max_value=2, value=1)
        seasonal_q = st.number_input("Seasonal MA Order (Q)", min_value=0, max_value=5, value=1)
        seasonal_m = st.number_input("Seasonal Period (M)", min_value=1, max_value=12, value=12)
        seasonal_order = (int(seasonal_p), int(seasonal_d), int(seasonal_q), int(seasonal_m))
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        try:
            df.columns = ["Month", "Sales"]
            df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
            df.dropna(subset=['Month'], inplace=True)
            df.set_index('Month', inplace=True)

            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<h3>Data Preview</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<h3>Time Series Visualization</h3>', unsafe_allow_html=True)
            fig = px.line(df, x=df.index, y='Sales', title="Time Series Visualization", 
                          labels={'Sales': 'Sales Values', 'Month': 'Time'})
            fig.update_traces(mode='lines+markers', hovertemplate='Time: %{x}<br>Sales: %{y}')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Generate Forecast"):
                with st.spinner('Training model and generating forecast...'):
                    model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(int(p), int(d), int(q)), seasonal_order=seasonal_order)
                    results = model.fit()

                    st.markdown('<div class="stCard">', unsafe_allow_html=True)
                    st.markdown('<h3>Model Results</h3>', unsafe_allow_html=True)
                    st.text(results.summary())
                    st.markdown('</div>', unsafe_allow_html=True)

                    future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1, 25)]
                    future_df = pd.DataFrame(index=future_dates, columns=df.columns)

                    forecast = results.predict(start=len(df), end=len(df) + 24, dynamic=True)
                    df['forecast'] = results.predict(start=0, end=len(df) - 1, dynamic=True)
                    future_df['forecast'] = forecast

                    combined_df = pd.concat([df[['Sales', 'forecast']], future_df[['forecast']]])

                    st.markdown('<div class="stCard">', unsafe_allow_html=True)
                    st.markdown('<h3>Forecast Results</h3>', unsafe_allow_html=True)
                    fig_forecast = px.line(
    combined_df,
    x=combined_df.index,
    y=['Sales', 'forecast'],
    title="Forecast Results",
    labels={'value': 'Sales Values', 'index': 'Time'},
)

    fig_forecast.update_traces(mode='lines+markers', hovertemplate='Time: %{x}<br>Value: %{y}')
    fig_forecast.update_layout(legend_title_text='Legend', legend=dict(x=0, y=1))

    st.plotly_chart(fig_forecast, use_container_width=True)


    except Exception as e:
    st.error(f"Error: {e}")

# Close main content
st.markdown('</div>', unsafe_allow_html=True)

# Fixed Footer
st.markdown("""
    <div class="footer">
        <div>Created by Akshitha Maddukuri | © 2025</div>
        <div class="footer-icons">
            <a href="https://github.com/akshithamaddukuri" target="_blank">
                <img src="https://img.icons8.com/ios-glyphs/30/ffffff/github.png" alt="GitHub">
            </a>
            <a href="https://www.linkedin.com/in/akshithamaddukuri524/" target="_blank">
                <img src="https://img.icons8.com/ios-filled/30/ffffff/linkedin.png" alt="LinkedIn">
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)