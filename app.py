import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st
import logging, warnings

logging.getLogger("fbprophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("📊 Factory-Wise Sales Forecast Dashboard")

uploaded_file = st.file_uploader("Upload your factory sales CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df.rename(columns={"Date": "ds", "Quantity_Sold": "y"}, inplace=True)

    latest_date = df["ds"].max().date()
    st.markdown(f"**🗓️ Latest Date in Dataset:** `{latest_date}`")

    summary = []
    for (product, factory), group in df.groupby(["Product_Name", "Factory"]):
        group = group.sort_values("ds")
        model = Prophet(daily_seasonality=True)
        model.fit(group[["ds", "y"]])
        forecast = model.make_future_dataframe(periods=30)
        forecast = model.predict(forecast)

        avg_hist = group["y"].mean()
        avg_pred = forecast["yhat"][-30:].mean()
        growth = ((avg_pred - avg_hist) / avg_hist * 100) if avg_hist else 0
        alert = "✅ Stable"
        if growth > 10: alert = "📈 Spike"
        elif growth < -10: alert = "📉 Drop"

        summary.append({
            "Product": product,
            "Factory": factory,
            "First_Date": group["ds"].min().date(),
            "Last_Date": group["ds"].max().date(),
            "Last_Sale": group['y'].iloc[-1],
            "Avg_Historical_Sales": round(avg_hist, 2),
            "Predicted_Avg_Sales": round(avg_pred, 2),
            "Total_Forecast_30d": round(forecast['yhat'][-30:].sum()),
            "Growth_%": round(growth, 2),
            "Alert": alert
        })

    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df)
else:
    st.info("📥 Upload a CSV file to begin.")
