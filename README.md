# 🚘 AutoValuate Pro | Enterprise Car Price Predictor

**AutoValuate Pro** is a Streamlit-powered Machine Learning Web Application designed to benchmark regression models and predict used car valuations based on vehicle specifications, condition, and location metrics.


## 📌 Features
- **Multi-Model Benchmarking**: Trains and evaluates **Linear Regression**, **Random Forest**, and **Gradient Boosting** models.
- **Live Metrics**: Displays real-time $R^2$ Score, MAE, and RMSE for each algorithm[cite: 1].
- **Dynamic UI**: Features custom dark-mode styling and automatic SVG body-type icons based on seating capacity[cite: 1].
- **ML Pipeline**: Uses Scikit-Learn `Pipeline` and `ColumnTransformer` with `OneHotEncoder` for feature processing[cite: 1].

---

## 📂 Project Structure
```text
├── car_app.py                            # Streamlit app & ML logic
├── car_details_expanded_14340_records.csv # Dataset
└── README.md                             # Documentation

Quick Start
Clone & Install Dependencies:

Bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install streamlit pandas numpy scikit-learn

Run Application:
Bash
python -m streamlit run car_app.py

📊 Model Input Parameters
Categorical: Brand, Fuel Type, Transmission, Owner, Seller Type, City, State[cite: 1].
Numerical: Year, Age, KM Driven, Engine CC, Mileage KMPL, Seats, Condition Score (1–5)[cite: 1].
python -m streamlit run car_app.py
