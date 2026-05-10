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
# TOP NAVIGATION TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Main Dashboard", 
    "📰 News Intelligence", 
    "📈 Forecasting Engine", 
    "🛠️ Scenario Simulator", 
    "ℹ️ About"
])

# =============================================================
# TAB 1: MAIN DASHBOARD (Phase 1 Focus)
# =============================================================
with tab1:
    st.markdown("### System Overview")
    st.info("Welcome to the Phase 1 Command Center. Use the tabs above to explore future Phase 2 modules.")
    
    col1, col2 = st.columns([1, 1])

    # --- COLUMN 1: NLP RISK SCORER ---
    with col1:
        st.header("📰 Geopolitical Risk Scorer")
        st.write("Test the NLP Model by entering a simulated news headline below:")
        
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
                
                st.markdown("#### 🧠 AI Reasoning: Why This Prediction?")
                st.write(f"*{result['Reasoning']}*")

    # --- COLUMN 2: LSTM PRICE PREDICTIONS ---
    with col2:
        st.header("📈 LSTM Price Prediction")
        st.write("30-Day Forecast based on 5 Years of Historical Data")
        
        last_30_history = historical_df.tail(30)
        last_date = historical_df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=last_30_history.index, 
            y=last_30_history['Price'],
            mode='lines+markers',
            name='Historical Prices (Past 30 Days)',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_predictions,
            mode='lines+markers',
            name='LSTM Forecast (Next 30 Days)',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="WTI Crude Oil Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($/Barrel)",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# =============================================================
# OTHER TABS (Phase 2 Placeholders)
# =============================================================
with tab2:
    st.markdown("### Live News Intelligence Feed")
    st.write("*(Phase 2 Integration)*")
    st.write("This module will display a continuous live stream of news from Reuters/GDELT, mapped automatically to risk scores and sentiment trends without manual input.")

with tab3:
    st.markdown("### Long-Term 3-5 Year Forecasting")
    st.write("*(Phase 2 Integration)*")
    st.write("This module will feature our Prophet/XGBoost models to predict multi-year structural trends in the Indian energy market.")

with tab4:
    st.markdown("### 🛠️ What-If Scenario Simulator")
    st.write("Test how hypothetical macroeconomic shocks alter the baseline LSTM trajectory.")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        st.markdown("#### Scenario Parameters")
        opec_cut = st.slider(
            "OPEC Production Change (Million BPD)", 
            min_value=-5.0, max_value=5.0, value=0.0, step=0.5, 
            help="Negative = production cut (price increases). Positive = oversupply (price drops)."
        )
        
        demand_shock = st.slider(
            "Global Demand Shock (%)", 
            min_value=-20, max_value=20, value=0, step=1, 
            help="Positive = economic boom/demand surge. Negative = recession/demand crash."
        )
        
        inr_usd = st.slider(
            "INR-USD Exchange Rate (₹)", 
            min_value=70.0, max_value=95.0, value=83.0, step=0.5, 
            help="Translates the global price into Indian Rupees for local impact."
        )
        
    with col_sim2:
        # Simple heuristic mapping for simulation:
        # 1M BPD cut = ~3% price increase. 1% demand shift = ~1.5% price shift.
        price_multiplier = 1.0 + (-opec_cut * 0.03) + (demand_shock * 0.015)
        scenario_predictions = [price * price_multiplier for price in future_predictions]
        
        fig_sim = go.Figure()
        
        # Base LSTM Forecast
        fig_sim.add_trace(go.Scatter(
            x=future_dates, 
            y=future_predictions,
            mode='lines',
            name='Base LSTM Forecast (USD)',
            line=dict(color='gray', dash='dash')
        ))
        
        # Scenario Forecast
        fig_sim.add_trace(go.Scatter(
            x=future_dates, 
            y=scenario_predictions,
            mode='lines+markers',
            name='Scenario Simulated Price (USD)',
            line=dict(color='orange', width=3)
        ))
        
        fig_sim.update_layout(
            title="Baseline vs. Shock Simulation",
            xaxis_title="Date",
            yaxis_title="Price ($/Barrel)",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Output Metrics
        avg_base = sum(future_predictions) / len(future_predictions)
        avg_scenario = sum(scenario_predictions) / len(scenario_predictions)
        
        st.markdown(f"**Average Base Price:** ${avg_base:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; **Average Scenario Price:** ${avg_scenario:.2f}")
        st.success(f"🇮🇳 **Estimated Indian Market Impact:** ₹{(avg_scenario * inr_usd):,.2f} per barrel")

with tab5:
    st.markdown("### About the Project")
    st.write("**GeoCognitive Energy Intelligence System**")
    st.write("Using Geopolitical and Macroeconomic Indicators.")
    st.markdown("---")
    st.write("**Team Members:** Kriti Agarwal, Shamit Sinha, Yash Agarwal, Raghav Somani")
    st.write("**Mentor:** Prof. Deepthi L")
    st.write("**Panel Member:** Dr. G S Nagaraja")

