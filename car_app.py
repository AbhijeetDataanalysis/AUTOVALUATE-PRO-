import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Page Configuration
st.set_page_config(
    page_title="AutoValuate AI | Car Price Predictor",
    page_icon="🚘",
    layout="wide",
)

# Custom Styling (Dark Automotive Theme, Badges, Visual Cards)
st.markdown(
    """
<style>
    /* Dark Theme Palette */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Hero Header Styling */
    .hero-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .hero-subtitle {
        color: #38bdf8;
        font-size: 1.05rem;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Vehicle Body Badge Box */
    .badge-card {
        background: #0f172a;
        border: 1px solid #38bdf8;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-top: 10px;
    }
    .badge-card svg {
        fill: #38bdf8;
        width: 110px;
        height: 55px;
    }

    /* Target Prediction Highlight */
    .pred-box {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border: 1px solid #34d399;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(52, 211, 153, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header Banner with Logo and Bold Title
st.markdown(
    """
<div class="hero-header">
    <div style="font-size: 50px;">🚘</div>
    <div>
        <div class="hero-title">AUTOVALUATE <span style="color: #38bdf8;">PRO</span></div>
        <div class="hero-subtitle">⚡ Enterprise Machine Learning Valuation & Multi-Model Benchmarking Engine</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)
    numeric_cols = [
        "year",
        "selling_price",
        "km_driven",
        "engine_cc",
        "mileage_kmpl",
        "seats",
        "car_age_years",
        "condition_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df


DATA_PATH = "car_details_expanded_14340_records.csv"
df = load_and_prep_data(DATA_PATH)

# Feature and Target Split
drop_cols = ["car_id", "name", "selling_price"]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df["selling_price"]

cat_features = X.select_dtypes(include=["object"]).columns.tolist()
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=42
    ),
}


@st.cache_resource
def train_models():
    trained_pipelines = {}
    metrics = {}
    for name, model in models.items():
        pipe = Pipeline(
            [("preprocessor", preprocessor), ("regressor", model)]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        trained_pipelines[name] = pipe
        metrics[name] = {
            "R² Score": r2_score(y_test, preds),
            "MAE (₹)": mean_absolute_error(y_test, preds),
            "RMSE (₹)": np.sqrt(mean_squared_error(y_test, preds)),
        }
    return trained_pipelines, metrics


pipelines, metrics = train_models()

# Section 1: Model Benchmark Leaderboard
st.markdown("### 📊 Model Performance Evaluation")
m_col1, m_col2, m_col3 = st.columns(3)
model_names = list(models.keys())

for idx, col in enumerate([m_col1, m_col2, m_col3]):
    m_name = model_names[idx]
    r2_val = metrics[m_name]["R² Score"]
    mae_val = metrics[m_name]["MAE (₹)"]
    col.metric(
        label=f"🤖 {m_name}",
        value=f"R²: {r2_val:.3f}",
        delta=f"MAE: ₹{mae_val:,.0f}",
        delta_color="inverse",
    )

st.markdown("---")

# Section 2: Vehicle Configuration & Visual Badge
st.markdown("### ⚙️ Vehicle Specifications")

c_input, c_visual = st.columns([3, 1])

with c_input:
    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("🏷️ Brand", sorted(df["brand"].unique()))
        year = st.slider(
            "📅 Manufacturing Year",
            int(df["year"].min()),
            int(df["year"].max()),
            2018,
        )
        car_age_years = st.slider("⌛ Vehicle Age (Years)", 0, 30, 2026 - year)
        fuel = st.selectbox("⛽ Fuel Type", sorted(df["fuel"].unique()))

    with col2:
        seller_type = st.selectbox(
            "🤝 Seller Type", sorted(df["seller_type"].unique())
        )
        transmission = st.selectbox(
            "⚙️ Transmission", sorted(df["transmission"].unique())
        )
        owner = st.selectbox("👤 Ownership History", sorted(df["owner"].unique()))
        km_driven = st.number_input(
            "🛣️ Odometer (KM)",
            min_value=0,
            max_value=500000,
            value=40000,
            step=1000,
        )

    with col3:
        engine_cc = st.number_input(
            "🏎️ Engine (CC)",
            min_value=500,
            max_value=5000,
            value=1200,
            step=50,
        )
        mileage_kmpl = st.number_input(
            "🛢️ Mileage (KMPL)",
            min_value=5.0,
            max_value=40.0,
            value=18.5,
            step=0.5,
        )
        seats = st.selectbox("💺 Seating Capacity", sorted(df["seats"].unique()))
        condition_score = st.slider(
            "⭐ Condition Rating (1-5)", 1.0, 5.0, 4.0, step=0.1
        )
        location = st.selectbox("📍 City", sorted(df["location"].unique()))
        state = st.selectbox("🗺️ State", sorted(df["state"].unique()))

# Visual Icon Badge Selection based on Seating Capacity
svg_icons = {
    "Hatchback / Sedan": '<svg viewBox="0 0 24 24"><path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.85 7h10.29l1.04 3H5.81l1.04-3zM19 17H5v-4h14v4z"/><circle cx="7.5" cy="15" r="1.5"/><circle cx="16.5" cy="15" r="1.5"/></svg>',
    "SUV / MUV": '<svg viewBox="0 0 24 24"><path d="M17 5H7c-1.1 0-2 .9-2 2v9c0 1.1.9 2 2 2h10c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 11H7V7h10v9z"/><circle cx="8.5" cy="13.5" r="1.5"/><circle cx="15.5" cy="13.5" r="1.5"/></svg>',
}

body_type = "Hatchback / Sedan" if seats <= 5 else "SUV / MUV"

with c_visual:
    st.markdown(
        f"""
    <div class="badge-card">
        <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">DETECTED BODY TYPE</div>
        <div style="margin: 10px 0;">{svg_icons[body_type]}</div>
        <div style="font-size: 1.1rem; color: #f8fafc; font-weight: 700;">{body_type}</div>
        <div style="font-size: 0.8rem; color: #38bdf8;">{seats} Seater Layout</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Section 3: Model Prediction Trigger
selected_model = st.selectbox(
    "🤖 Select Prediction Model", list(models.keys()), index=1
)

if st.button("🚀 Calculate Estimated Market Value", use_container_width=True):
    input_data = pd.DataFrame(
        [
            {
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
                "brand": brand,
                "engine_cc": engine_cc,
                "mileage_kmpl": mileage_kmpl,
                "seats": seats,
                "location": location,
                "state": state,
                "car_age_years": car_age_years,
                "condition_score": condition_score,
            }
        ]
    )

    predicted_price = pipelines[selected_model].predict(input_data)[0]

    st.markdown(
        f"""
    <div class="pred-box">
        <div>Estimated Market Price ({selected_model})</div>
        <div style="font-size: 2.8rem; font-weight: 800; margin-top: 5px;">₹{predicted_price:,.2f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )