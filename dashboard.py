import streamlit as st
from robo_advisor.core import RoboAdvisor

def main():
    st.title("AI Robo-Advisor Dashboard")
    
    # User inputs
    age = st.slider("Age", 18, 100, 30)
    income = st.number_input("Annual Income ($)", 30000, 1000000, 75000)
    savings = st.number_input("Savings ($)", 0, 10000000, 100000)
    risk_tolerance = st.select_slider("Risk Tolerance (1 = Conservative, 2 = Moderate, 3 = Aggressive)", options=[1, 2, 3], value=2)
    investment_horizon = st.selectbox("Investment Horizon (In Years)", [1, 3, 5, 10])
    
    # Generate recommendation
    if st.button("Generate Portfolio"):
        advisor = RoboAdvisor()
        user_data = {
            'age': age,
            'income': income,
            'savings': savings,
            'risk_tolerance': risk_tolerance,
            'investment_horizon': investment_horizon
        }
        
        portfolio = advisor.generate_portfolio(user_data)
        
        # Display results
        st.subheader("Recommended Portfolio")
        st.write(f"Risk Profile: {portfolio['risk_profile']}")
        
        st.subheader("Asset Allocation")
        for asset, alloc in portfolio['allocation'].items():
            st.progress(alloc)
            st.write(f"{asset}: {alloc:.1%}")
            
        st.subheader("Performance Metrics")
        st.metric("Expected Return", f"{portfolio['statistics']['expected_return']:.1%}")
        st.metric("Volatility", f"{portfolio['statistics']['volatility']:.1%}")
        st.metric("Sharpe Ratio", f"{portfolio['statistics']['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    main() 