"""M3-owned Streamlit UI. Reads model artifacts from M2's documented contract."""
import logging
from pathlib import Path
import streamlit as st
from src.shared.config import load_config, setup_logging
from src.modeling.inference import validate_car, predict_price

LOG = logging.getLogger(__name__)

def main():
    """Render the form, predict, and show a saved EDA figure."""
    setup_logging()
    st.set_page_config(page_title="Used Car Price Predictor", layout="wide")
    st.title("Craigslist Used Car Price Predictor")
    st.caption("Illustrative estimate of a listing price in USD, not a confirmed sale price.")
    try:
        cfg = load_config()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    with st.form("car_specs"):
        left, right = st.columns(2)
        with left:
            manufacturer = st.text_input("Manufacturer", value="toyota")
            year = st.number_input("Model year", min_value=1900, max_value=2100, value=2018, step=1)
            odometer = st.number_input("Odometer (miles)", min_value=-100000, max_value=1000000, value=65000, step=1000)
        with right:
            fuel = st.selectbox("Fuel", ["gas", "diesel", "hybrid", "electric", "other"])
            transmission = st.selectbox("Transmission", ["automatic", "manual", "other"])
        submitted = st.form_submit_button("Predict price")
    if submitted:
        try:
            car = validate_car(int(year), float(odometer), manufacturer, fuel, transmission, cfg)
            amount = predict_price(cfg["paths"]["model"], car)
            st.metric("Estimated listing price (USD)", f"${amount:,.0f}")
        except (ValueError, FileNotFoundError) as exc:
            st.warning(str(exc))
        except Exception:
            LOG.exception("Unexpected prediction error")
            st.error("Prediction failed. See application logs for details.")
    st.subheader("Exploratory data analysis")
    fig = Path(cfg["paths"]["chart"])
    if fig.exists():
        st.image(str(fig), caption="Distribution of cleaned Craigslist listing prices")
    else:
        st.info("Run the data processing command to create the EDA visualization.")

if __name__ == "__main__":
    main()
