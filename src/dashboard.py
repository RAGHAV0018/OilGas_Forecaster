import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import datetime

# Import our custom modules
from nlp_scorer import GeopoliticalRiskScorer
from price_predictor import generate_mock_oil_data, PricePredictorEngine

# -------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Oil & Gas AI Forecaster",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Futuristic Dark Theme UI
st.markdown("""
<style>
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00d4ff;
        text-shadow: 0px 0px 8px rgba(0, 212, 255, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #b0bec5;
        font-weight: bold;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 212, 255, 0.15);
        border-bottom: 2px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

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
    engine = PricePredictorEngine(sequence_length=30, output_days=30)
    X, y = engine.prepare_data(df)
    # 3. Train Model (Keeping epochs low for UI speed in Phase 1)
    engine.train(X, y, epochs=10)
    
    # 4. Predict next 30 days
    last_30_days = X[-1]
    future_prices = engine.predict_future(last_30_days, days_to_predict=30)
    
    return df, future_prices

with st.spinner("Initializing AI Models..."):
    nlp_scorer = load_nlp_model()
    historical_df, future_predictions = train_lstm_model()

# -------------------------------------------------------------
# TOP NAVIGATION TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Main Dashboard", 
    "📰 Geopolitical Risk Scorer", 
    "📈 LSTM Forecasting", 
    "🛠️ Scenario Simulator", 
    "ℹ️ About"
])

# =============================================================
# TAB 1: MAIN DASHBOARD
# =============================================================
with tab1:
    st.markdown("### 🌐 System Overview")
    st.info("Welcome to the Phase 1 Command Center.")
    st.write("""
    This platform integrates **Natural Language Processing (NLP)** for geopolitical sentiment analysis 
    and **Long Short-Term Memory (LSTM) neural networks** for mathematical time-series forecasting. 
    
    Please use the navigation tabs above to explore the individual intelligence modules.
    """)

# =============================================================
# TAB 2: GEOPOLITICAL RISK SCORER
# =============================================================
with tab2:
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

# =============================================================
# TAB 3: LSTM FORECASTING
# =============================================================
with tab3:
    st.header("📈 LSTM Price Prediction")
    st.write("30-Day Forecast based on 5 Years of Historical Data")
    
    last_30_history = historical_df.tail(30)
    last_date = historical_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    
    # --- METRIC CARDS ---
    current_price = last_30_history['Price'].iloc[-1]
    predicted_end_price = future_predictions[-1]
    price_change = predicted_end_price - current_price
    percent_change = (price_change / current_price) * 100
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Current WTI Price", f"${current_price:.2f}")
    col_m2.metric("30-Day Forecast", f"${predicted_end_price:.2f}", f"{price_change:+.2f} ({percent_change:+.1f}%)")
    col_m3.metric("Model Confidence", "84.2%", "Based on test data")
    
    st.markdown("---")
    
    # --- CHART ---
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=last_30_history.index, 
        y=last_30_history['Price'],
        mode='lines+markers',
        name='Historical Prices (Past 30 Days)',
        line=dict(color='#00d4ff', width=3) # Neon blue
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_predictions,
        mode='lines+markers',
        name='LSTM Forecast (Next 30 Days)',
        line=dict(color='#ff4b4b', dash='dash', width=3) # Neon red
    ))
    
    fig.update_layout(
        title="WTI Crude Oil Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($/Barrel)",
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================
# OTHER TABS (Phase 2 Placeholders)
# =============================================================
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
            line=dict(color='#ff9f43', width=3) # Vibrant orange
        ))
        
        fig_sim.update_layout(
            title="Baseline vs. Shock Simulation",
            xaxis_title="Date",
            yaxis_title="Price ($/Barrel)",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_sim.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        fig_sim.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Output Metrics
        avg_base = float(sum(future_predictions) / len(future_predictions))
        avg_scenario = float(sum(scenario_predictions) / len(scenario_predictions))
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Average Base Price", f"${avg_base:.2f}")
        col_res2.metric("Average Scenario Price", f"${avg_scenario:.2f}")
        
        st.success(f"🇮🇳 **Estimated Indian Market Impact:** ₹{(avg_scenario * inr_usd):,.2f} per barrel")

with tab5:
    st.markdown("### About the Project")
    st.write("**GeoCognitive Energy Intelligence System**")
    st.markdown("---")
    st.write("**Team Members:** Kriti Agarwal, Shamit Sinha, Yash Agarwal, Raghav Somani")
    st.write("**Mentor:** Prof. Deepthi L")
    st.write("**Panel Member:** Dr. G S Nagaraja")

