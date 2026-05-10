import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import datetime

# Import our custom modules
from nlp_scorer import GeopoliticalRiskScorer
from price_predictor import generate_mock_oil_data, PricePredictorEngine

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Oil & Gas AI Forecaster",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("🛢️ GeoCognitive Energy Intelligence System")

# Initialize models only once using Streamlit caching
@st.cache_resource
def load_nlp_model():
    return GeopoliticalRiskScorer()

@st.cache_resource
def train_lstm_model():
    # 1. Load Data
    df = generate_mock_oil_data(days=500)
    # 2. Setup Engine
    engine = PricePredictorEngine(sequence_length=30)
    X, y = engine.prepare_data(df)
    # 3. Train Model (Keeping epochs low for UI speed in Phase 1)
    engine.train(X, y, epochs=10)
    
    # 4. Predict next 30 days
    last_30_days = X[-1]
    future_prices = engine.predict_future(last_30_days, days_to_predict=30)
    
    return df, future_prices

with st.spinner("Loading AI Models... (This might take a few seconds on first run)"):
    nlp_scorer = load_nlp_model()
    historical_df, future_predictions = train_lstm_model()

# -------------------------------------------------------------
# UI LAYOUT: TWO COLUMNS
# -------------------------------------------------------------
col1, col2 = st.columns([1, 1])

# --- COLUMN 1: NLP RISK SCORER ---
with col1:
    st.header("📰 Geopolitical Risk Scorer")
    st.write("Test the NLP Model by entering a simulated news headline below:")
    
    # Text input for user
    user_headline = st.text_area(
        "Enter News Headline:",
        "OPEC announces unexpected cut to oil production by 2 million barrels per day, citing market instability."
    )
    
    if st.button("Analyze Impact"):
        with st.spinner("Analyzing text with FinBERT..."):
            result = nlp_scorer.analyze_headline(user_headline)
            
            st.markdown("### Analysis Results:")
            st.info(f"**Detected Sentiment:** {result['Raw_Sentiment']} (Confidence: {result['Confidence']}%)")
            
            if "HIGH RISK" in result['Geopolitical_Impact']:
                st.error(f"**Geopolitical Impact:** {result['Geopolitical_Impact']}")
            else:
                st.success(f"**Geopolitical Impact:** {result['Geopolitical_Impact']}")
                
            st.warning(f"**Predicted Price Trend:** {result['Predicted_Price_Trend']}")

# --- COLUMN 2: LSTM PRICE PREDICTIONS ---
with col2:
    st.header("📈 LSTM Price Prediction")
    st.write("30-Day Forecast based on Historical Data Patterns (Phase 1 Base Model)")
    
    # Prepare data for plotting
    last_30_history = historical_df.tail(30)
    
    # Generate future dates
    last_date = historical_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    
    # Create Plotly Graph
    fig = go.Figure()
    
    # Plot historical
    fig.add_trace(go.Scatter(
        x=last_30_history.index, 
        y=last_30_history['Price'],
        mode='lines+markers',
        name='Historical Prices (Past 30 Days)',
        line=dict(color='blue')
    ))
    
    # Plot predicted
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_predictions,
        mode='lines+markers',
        name='LSTM Forecast (Next 30 Days)',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Oil Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($/Barrel)",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Display Graph
    st.plotly_chart(fig, use_container_width=True)

