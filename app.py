"""Used Car Price Prediction - Streamlit web app.
Loads the model trained in the notebook and predicts the selling price (INR)."""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

MODEL_FILE = "used_car_price_model.pkl"
INFO_FILE = "model_info.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_info():
    with open(INFO_FILE, encoding="utf-8") as f:
        return json.load(f)


model = load_model()
info = load_info()

st.title("🚗 Used Car Price Predictor")
st.caption(
    f"Model: **{info['best_algorithm']}**, chosen from 5 algorithms using "
    f"{info['cv_folds']}-fold cross validation on {info['n_cars']:,} cars."
)

tab_predict, tab_compare = st.tabs(["Predict price", "Algorithm comparison"])

with tab_predict:
    c1, c2 = st.columns(2)
    brand = c1.selectbox("Brand", info["brands"])
    year = c2.number_input("Year of manufacture", info["year_min"], info["year_max"] + 1, 2017, 1)
    km = c1.number_input("Kilometres driven", 0, 1_000_000, 60_000, 1000)
    fuel = c2.selectbox("Fuel type", info["fuels"])
    transmission = c1.selectbox("Transmission", info["transmissions"])
    seller = c2.selectbox("Seller type", info["seller_types"])
    owner = st.selectbox("Owner", info["owners"])

    st.markdown("**Optional details** (leave the box unticked if you don't know them)")
    o1, o2, o3 = st.columns(3)
    engine = o1.number_input("Engine (CC)", 500, 7000, 1200, 50) if o1.checkbox("I know the engine CC") else np.nan
    power = o2.number_input("Power (bhp)", 30.0, 700.0, 85.0, 1.0) if o2.checkbox("I know the power") else np.nan
    seats = o3.number_input("Seats", 2, 14, 5, 1) if o3.checkbox("I know the seats") else np.nan

    if st.button("Predict price", type="primary"):
        row = pd.DataFrame([{
            "year": year, "km_driven": km, "engine_cc": engine, "max_power_bhp": power, "seats": seats,
            "brand": brand, "fuel": fuel, "seller_type": seller, "transmission": transmission, "owner": owner,
        }])
        price = max(float(model.predict(row)[0]), 0.0)
        st.success(f"Estimated selling price: **₹{price:,.0f}**  ({price / 1e5:,.2f} lakh)")
        st.caption(
            f"Typical error on ordinary cars (MAE) is about ₹{info['mean_mae']:,.0f}; "
            f"cross-validation RMSE is ₹{info['mean_rmse']:,.0f} (inflated by a few luxury cars)."
        )

with tab_compare:
    st.write("10-fold cross-validation results (lower RMSE is better):")
    table = pd.DataFrame(info["comparison"]).set_index("Algorithm")
    st.dataframe(table.style.format({
        "Mean RMSE (₹)": "₹{:,.0f}", "Std RMSE (₹)": "₹{:,.0f}", "Mean MAE (₹)": "₹{:,.0f}",
        "Mean R²": "{:.3f}", "Time (s)": "{:.1f}"}))
    st.bar_chart(table["Mean RMSE (₹)"])
