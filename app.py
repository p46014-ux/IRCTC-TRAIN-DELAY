import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="ABC Ltd. | Journey Delay Prediction",
    page_icon="🚆",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

logistic_model = joblib.load("logistic_delay_model.pkl")
linear_model = joblib.load("linear_delay_model.pkl")
model_info = joblib.load("model_info.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🚆 ABC Ltd. — Journey Delay Prediction")
st.markdown(
    "### Predict delay risk and estimated delay duration before departure"
)

st.divider()

# --------------------------------------------------
# INPUTS
# --------------------------------------------------

st.subheader("Enter Journey Details")

col1, col2, col3 = st.columns(3)

with col1:
    train_number = st.number_input(
        "Train Number",
        min_value=1,
        value=12345,
        step=1
    )

    train_type = st.selectbox(
        "Train Type",
        logistic_model.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["onehot"]
        .categories_[1]
    )

    departure_date = st.date_input(
        "Departure Date",
        value=date.today()
    )

    departure_hour = st.slider(
        "Departure Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    season = st.selectbox(
        "Season",
        logistic_model.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["onehot"]
        .categories_[2]
    )

with col2:
    zone = st.selectbox(
        "Zone",
        logistic_model.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["onehot"]
        .categories_[3]
    )

    zone_abbr = st.selectbox(
        "Zone Abbreviation",
        logistic_model.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["onehot"]
        .categories_[4]
    )

    source_station_category = st.selectbox(
        "Source Station Category",
        logistic_model.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["onehot"]
        .categories_[5]
    )

    destination_station_category = st.selectbox(
        "Destination Station Category",
        logistic_model.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["onehot"]
        .categories_[6]
    )

    traction_type = st.selectbox(
        "Traction Type",
        logistic_model.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["onehot"]
        .categories_[7]
    )

with col3:
    distance_km = st.number_input(
        "Distance (km)",
        min_value=0.0,
        value=500.0
    )

    num_scheduled_stops = st.number_input(
        "Scheduled Stops",
        min_value=0,
        value=10,
        step=1
    )

    scheduled_travel_hours = st.number_input(
        "Scheduled Travel Hours",
        min_value=0.0,
        value=8.0
    )

    psr_count = st.number_input(
        "PSR Count",
        min_value=0,
        value=0,
        step=1
    )

    maintenance_score = st.number_input(
        "Maintenance Score",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

# --------------------------------------------------
# OPERATIONAL VARIABLES
# --------------------------------------------------

st.subheader("Operational Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    fog_risk_score = st.number_input(
        "Fog Risk Score",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

    zone_fog_index = st.number_input(
        "Zone Fog Index",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

    zone_congestion_index = st.number_input(
        "Zone Congestion Index",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    season_severity_score = st.number_input(
        "Season Severity Score",
        min_value=0.0,
        max_value=100.0,
        value=30.0
    )

with col2:
    loco_age_years = st.number_input(
        "Loco Age (years)",
        min_value=0.0,
        value=5.0
    )

    coach_age_years = st.number_input(
        "Coach Age (years)",
        min_value=0.0,
        value=5.0
    )

    seat_utilisation_pct = st.number_input(
        "Seat Utilisation (%)",
        min_value=0.0,
        max_value=200.0,
        value=80.0
    )

    route_historical_ontime_pct = st.number_input(
        "Historical On-time (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

with col3:
    track_doubled = st.selectbox(
        "Track Doubled?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_hdn_route = st.selectbox(
        "High Density Route?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_electrified = st.selectbox(
        "Electrified?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    late_incoming_rake = st.selectbox(
        "Incoming Rake Already Late?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# --------------------------------------------------
# ADDITIONAL CONDITIONS
# --------------------------------------------------

st.subheader("Additional Conditions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    is_weekend = int(departure_date.weekday() >= 5)
    st.metric("Weekend", "Yes" if is_weekend else "No")

with col2:
    is_night_departure = int(
        departure_hour >= 22 or departure_hour < 6
    )
    st.metric(
        "Night Departure",
        "Yes" if is_night_departure else "No"
    )

with col3:
    is_peak_hour = st.selectbox(
        "Peak Hour?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col4:
    is_festival_season = st.selectbox(
        "Festival Season?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button("🔮 Predict Journey Delay", type="primary"):

    departure_day = departure_date.day
    departure_weekofyear = departure_date.isocalendar().week

    input_data = pd.DataFrame([{
        "train_number": train_number,
        "train_type": train_type,
        "year": departure_date.year,
        "month": departure_date.month,
        "day_of_week": departure_date.weekday(),
        "departure_hour": departure_hour,
        "is_weekend": is_weekend,
        "is_night_departure": is_night_departure,
        "is_peak_hour": is_peak_hour,
        "is_festival_season": is_festival_season,
        "season": season,
        "zone": zone,
        "zone_abbr": zone_abbr,
        "source_station_category": source_station_category,
        "destination_station_category": destination_station_category,
        "distance_km": distance_km,
        "num_scheduled_stops": num_scheduled_stops,
        "scheduled_travel_hours": scheduled_travel_hours,
        "track_doubled": track_doubled,
        "is_hdn_route": is_hdn_route,
        "traction_type": traction_type,
        "is_electrified": is_electrified,
        "psr_count": psr_count,
        "is_circular_route": 0,
        "is_monsoon_season": 0,
        "is_fog_risk": int(fog_risk_score > 50),
        "fog_risk_score": fog_risk_score,
        "zone_fog_index": zone_fog_index,
        "zone_congestion_index": zone_congestion_index,
        "season_severity_score": season_severity_score,
        "loco_age_years": loco_age_years,
        "coach_age_years": coach_age_years,
        "has_lhb_coaches": 1,
        "is_rake_shared": 0,
        "maintenance_score": maintenance_score,
        "seat_utilisation_pct": seat_utilisation_pct,
        "is_overloaded": int(seat_utilisation_pct > 100),
        "late_incoming_rake": late_incoming_rake,
        "is_special_train": 0,
        "route_historical_ontime_pct": route_historical_ontime_pct,
        "departure_day": departure_day,
        "departure_weekofyear": float(departure_weekofyear)
    }])

    # Logistic prediction
    delay_probability = logistic_model.predict_proba(
        input_data
    )[0][1]

    delay_prediction = logistic_model.predict(
        input_data
    )[0]

    st.subheader("Prediction Result")

    if delay_prediction == 1:

        st.error("🔴 HIGH DELAY RISK")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Probability of Delay",
                f"{delay_probability * 100:.1f}%"
            )

        # Linear regression
        estimated_delay = linear_model.predict(
            input_data
        )[0]

        estimated_delay = max(0, estimated_delay)

        with col2:
            st.metric(
                "Estimated Delay",
                f"{estimated_delay:.0f} minutes"
            )

        st.warning(
            "The model predicts that this journey is likely "
            "to exceed the 15-minute delay threshold."
        )

    else:

        st.success("🟢 LOW DELAY RISK")

        st.metric(
            "Probability of Delay",
            f"{delay_probability * 100:.1f}%"
        )

        st.info(
            "The model predicts that this journey is unlikely "
            "to exceed the 15-minute delay threshold."
        )

    st.caption(
        "Prediction generated using Logistic Regression for delay "
        "classification and Linear Regression for expected delay duration."
    )
