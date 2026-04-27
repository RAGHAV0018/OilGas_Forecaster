import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline

class GeopoliticalRiskScorer:
    def __init__(self):
        print("Loading Financial NLP Model (FinBERT)...")
        # We use FinBERT which is specifically trained on financial text
        # It classifies text into: Positive, Negative, Neutral
        # In oil markets, "Negative" supply news often means prices will go UP.
        self.nlp_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
        print("Model loaded successfully.\n")

    def analyze_headline(self, headline):
        """
        Analyzes a news headline and returns a market impact prediction.
        """
        result = self.nlp_pipeline(headline)[0]
        sentiment = result['label']
        confidence = result['score']
        
        # Translating Financial Sentiment to Oil Price Impact
        # If the financial sentiment is negative (e.g. "war breaks out", "supply drops"), 
        # it usually means oil prices will surge.
        if sentiment == 'negative':
            impact = "HIGH RISK / SUPPLY SHOCK"
            price_trend = "UPWARDS (Bullish)"
        elif sentiment == 'positive':
            impact = "STABLE / OVERSUPPLY"
            price_trend = "DOWNWARDS (Bearish)"
        else:
            impact = "NEUTRAL"
            price_trend = "STABLE"

        return {
            "Headline": headline,
            "Raw_Sentiment": sentiment.upper(),
            "Confidence": round(confidence * 100, 2),
            "Geopolitical_Impact": impact,
            "Predicted_Price_Trend": price_trend
        }

if __name__ == "__main__":
    scorer = GeopoliticalRiskScorer()
    
    # Test cases for Phase 1 Demonstration
    test_headlines = [
        "OPEC announces unexpected cut to oil production by 2 million barrels per day.",
        "New peace treaty signed in the Middle East, ensuring safe passage in Strait of Hormuz.",
        "India announces plans to boost electric vehicle subsidies."
    ]
    
    for news in test_headlines:
        print("-" * 50)
        print(f"NEWS HEADLINE: {news}")
        analysis = scorer.analyze_headline(news)
        print(f"Geopolitical Impact : {analysis['Geopolitical_Impact']}")
        print(f"Price Trend         : {analysis['Predicted_Price_Trend']}")
        print(f"Confidence Level    : {analysis['Confidence']}%")
