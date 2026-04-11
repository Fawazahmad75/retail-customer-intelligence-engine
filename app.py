import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Retail Customer Intelligence Engine",
    page_icon="🛒",
    layout="wide"
)

# ============ LOAD MODELS ============
@st.cache_resource
def load_models():
    churn_model = joblib.load('models/churn_rf.joblib')
    text_model = joblib.load('models/text_classifier_linerR.joblib')
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    return churn_model, text_model, vectorizer

churn_model, text_model, vectorizer = load_models()
def calculate_rfm_segment(tenure, days_since, order_count, cashback):
    """Calculate RFM segment based on customer behavior"""
    if days_since <= 2: r = 5
    elif days_since <= 5: r = 4
    elif days_since <= 10: r = 3
    elif days_since <= 20: r = 2
    else: r = 1
    
    if order_count >= 15: f = 5
    elif order_count >= 10: f = 4
    elif order_count >= 5: f = 3
    elif order_count >= 2: f = 2
    else: f = 1
    
    if cashback >= 250: m = 5
    elif cashback >= 200: m = 4
    elif cashback >= 150: m = 3
    elif cashback >= 100: m = 2
    else: m = 1
    
    score = r + f + m
    
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions', score
    elif f >= 3 and m >= 3:
        return 'Loyal', score
    elif r >= 4 and f <= 2:
        return 'New Customers', score
    elif r <= 2 and f >= 3:
        return 'At Risk', score
    elif r <= 2 and f <= 2:
        return 'Lost', score
    else:
        return 'Needs Attention', score

def calculate_clv(cashback, order_count, tenure):
    """Estimate customer lifetime value"""
    return cashback * order_count * (tenure / 12)

# ============ HEADER ============
st.title("🛒 Retail Customer Intelligence Engine")
st.markdown("""
**An end-to-end AI system that predicts customer churn and classifies feedback sentiment.**
Built as part of my application for the Junior Applied AI Engineer role at Genesis AI Garage.
Inspired by my experience working at Walmart, where I see firsthand how customer behavior
and feedback signal retention risk.
""")

# ============ TABS ============
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Churn Predictor", "💬 Review Classifier", "🔮 Full Analysis", "📊 About"])
# ============ TAB 1: CHURN PREDICTOR ============
with tab1:
    st.header("Customer Churn Prediction")
    st.write("Enter customer details to predict their likelihood of churning.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 60, 12)
        satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
        complain = st.selectbox("Filed a Complaint?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        days_since = st.number_input("Days Since Last Order", 0, 60, 5)
        order_count = st.number_input("Total Orders", 1, 50, 5)
        cashback = st.number_input("Cashback Amount ($)", 0.0, 500.0, 150.0)
        warehouse_distance = st.number_input("Warehouse to Home (km)", 0, 200, 15)
    
    with col2:
        num_addresses = st.number_input("Number of Addresses", 1, 20, 3)
        num_devices = st.number_input("Devices Registered", 1, 10, 3)
        hour_on_app = st.number_input("Hours Spent on App", 0, 10, 3)
        coupon_used = st.number_input("Coupons Used", 0, 20, 2)
        order_hike = st.number_input("Order Hike From Last Year (%)", 0, 100, 15)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
    
    if st.button("🔮 Predict Churn Risk", type="primary"):
        # Build feature array — must match training feature order
        features = np.zeros(29)
        features[0] = tenure
        features[1] = city_tier
        features[2] = warehouse_distance
        features[3] = hour_on_app
        features[4] = num_devices
        features[5] = satisfaction
        features[6] = num_addresses
        features[7] = complain
        features[8] = order_hike
        features[9] = coupon_used
        features[10] = order_count
        features[11] = days_since
        features[12] = cashback
        # Remaining features (one-hot encoded) default to 0
        
        prediction = churn_model.predict([features])[0]
        probability = churn_model.predict_proba([features])[0][1]
        
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Churn Probability", f"{probability:.1%}")
        with col_b:
            risk = "🔴 HIGH" if probability > 0.7 else "🟡 MEDIUM" if probability > 0.4 else "🟢 LOW"
            st.metric("Risk Level", risk)
        with col_c:
            st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")
        
        if probability > 0.5:
            st.error(f"⚠️ This customer is at high risk of churning. Recommended actions: targeted retention offer, personal outreach, or satisfaction survey.")
        else:
            st.success(f"✅ This customer appears stable. Maintain engagement with regular communications.")

# ============ TAB 2: REVIEW CLASSIFIER ============
with tab2:
    st.header("Customer Review Sentiment Classifier")
    st.write("Paste a customer review to classify it as positive or negative.")
    
    review_text = st.text_area(
        "Customer Review:",
        placeholder="Example: 'This product was disappointing, the size was wrong and delivery was slow.'",
        height=150
    )
    
    if st.button("🔍 Classify Review", type="primary"):
        if review_text.strip():
            review_tfidf = vectorizer.transform([review_text])
            prediction = text_model.predict(review_tfidf)[0]
            probability = text_model.predict_proba(review_tfidf)[0]
            
            st.markdown("---")
            col_a, col_b = st.columns(2)
            
            with col_a:
               confidence = max(probability)
               if confidence < 0.65:
                st.warning("⚠️ UNCERTAIN — Needs Human Review")
                st.caption("The model isn't confident about this review. In production, this would be routed to a human agent.")
               elif prediction == 1:
                st.success("✅ POSITIVE Review")
               else:
                st.error("❌ NEGATIVE Review")
            
            with col_b:
                confidence = max(probability)
                st.metric("Confidence", f"{confidence:.1%}")
            
            if prediction == 0:
                st.warning("⚠️ This is a negative review. In a real business, this would trigger an alert for customer service follow-up.")
        else:
            st.warning("Please enter a review to classify.")

# ============ TAB 3: FULL ANALYSIS ============
with tab3:
    st.header("Complete Customer Intelligence Report")
    st.write("Combine behavioral data AND customer feedback for the full picture.")
    
    st.subheader("Customer Behavior Data")
    fcol1, fcol2 = st.columns(2)
    
    with fcol1:
        f_tenure = st.slider("Tenure (months)", 0, 60, 12, key="f_tenure")
        f_satisfaction = st.slider("Satisfaction Score", 1, 5, 3, key="f_sat")
        f_complain = st.selectbox("Filed Complaint?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="f_comp")
        f_days = st.number_input("Days Since Last Order", 0, 60, 5, key="f_days")
    
    with fcol2:
        f_orders = st.number_input("Total Orders", 1, 50, 5, key="f_ord")
        f_cashback = st.number_input("Cashback ($)", 0.0, 500.0, 150.0, key="f_cash")
        f_addresses = st.number_input("Number of Addresses", 1, 20, 3, key="f_addr")
        f_city = st.selectbox("City Tier", [1, 2, 3], key="f_city")
    
    st.subheader("Customer Feedback")
    f_review = st.text_area(
        "Latest customer review or feedback:",
        placeholder="Paste the customer's most recent review or support message...",
        height=100,
        key="f_review"
    )
    
    if st.button("🚀 Generate Full Intelligence Report", type="primary"):
        # Build features
        features = np.zeros(29)
        features[0] = f_tenure
        features[1] = f_city
        features[5] = f_satisfaction
        features[6] = f_addresses
        features[7] = f_complain
        features[10] = f_orders
        features[11] = f_days
        features[12] = f_cashback
        features[13] = 1  # PreferredLoginDevice_Mobile Phone (most common)
        features[18] = 1  # PreferredPaymentMode_Debit Card (common)
        features[24] = 1  # PreferredOrderCat_Mobile (common)
        features[28] = 1  # MaritalStatus_Single
        
        # Run all models
        churn_pred = churn_model.predict([features])[0]
        churn_prob = churn_model.predict_proba([features])[0][1]
        segment, rfm_score = calculate_rfm_segment(f_tenure, f_days, f_orders, f_cashback)
        clv = calculate_clv(f_cashback, f_orders, f_tenure)
        
        # Sentiment analysis
        sentiment_label = "Not provided"
        sentiment_confidence = 0
        if f_review.strip():
            review_tfidf = vectorizer.transform([f_review])
            sent_pred = text_model.predict(review_tfidf)[0]
            sent_proba = text_model.predict_proba(review_tfidf)[0]
            sentiment_confidence = max(sent_proba)
            
            if sentiment_confidence < 0.65:
                sentiment_label = "Uncertain"
            elif sent_pred == 1:
                sentiment_label = "Positive"
            else:
                sentiment_label = "Negative"
        
        st.markdown("---")
        st.subheader("📋 Intelligence Report")
        
        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Churn Risk", f"{churn_prob:.1%}")
        m2.metric("Segment", segment)
        m3.metric("Lifetime Value", f"${clv:,.0f}")
        m4.metric("Review Sentiment", sentiment_label)
        
        # Combined insight
        st.markdown("### 🎯 Combined Insight")
        
        risk_high = churn_prob > 0.5
        sentiment_negative = sentiment_label == "Negative"
        
        if risk_high and sentiment_negative:
            st.error(f"""
            **🚨 CRITICAL: Immediate intervention needed**
            
            This is a {segment} customer with {churn_prob:.0%} churn risk AND negative feedback.
            Their behavior signals match what they're saying — they're actively unhappy and likely to leave.
            
            **Recommended action:** Personal outreach within 24 hours. Address their specific complaint
            and offer a meaningful retention incentive. Estimated revenue at risk: ${clv:,.2f}
            """)
        elif risk_high and not sentiment_negative:
            st.warning(f"""
            **⚠️ Behavioral risk detected — feedback doesn't show it yet**
            
            This {segment} customer shows {churn_prob:.0%} churn risk based on behavior patterns,
            but their feedback isn't negative. They may be quietly disengaging.
            
            **Recommended action:** Proactive outreach with personalized offer. Survey to identify hidden issues.
            """)
        elif not risk_high and sentiment_negative:
            st.warning(f"""
            **💬 Vocal complaint, but behavior is stable**
            
            This {segment} customer is venting frustration but their purchase behavior is healthy.
            They may just need acknowledgment and a quick fix.
            
            **Recommended action:** Respond to feedback promptly. A small gesture goes a long way.
            """)
        else:
            st.success(f"""
            **✅ Healthy customer**
            
            This {segment} customer shows low churn risk and positive/neutral feedback.
            Lifetime value: ${clv:,.2f}
            
            **Recommended action:** Maintain engagement. Consider for loyalty program if Champion/Loyal segment.
            """)

# ============ TAB 4: ABOUT ============
with tab4:
    st.header("About This Project")
    st.markdown("""
    ### What This Is
    A complete AI system combining two models:
    
    **1. Churn Prediction Model** — Random Forest trained on 5,630 e-commerce customers
    - Test Accuracy: 98.3% | F1: 0.95 | AUC: 0.999
    - Compared against Logistic Regression, Gradient Boosting, and a custom PyTorch neural network
    
    **2. Sentiment Classifier** — Logistic Regression with TF-IDF on 22,641 customer reviews
    - Catches 82% of negative reviews (vs 33% for Random Forest baseline)
    - Demonstrates business-aware metric selection
    
    ### Tech Stack
    Python, scikit-learn, PyTorch, SHAP, pandas, NumPy, Streamlit
    
    ### My Walmart Connection
    Working at Walmart, I see firsthand how customers leave for predictable reasons — 
    a bad pickup experience, a slow delivery, an unresolved complaint. This system 
    automates pattern recognition that experienced retail workers do intuitively.
    
    ### Built By
    Fawaz Ahmad
    """)