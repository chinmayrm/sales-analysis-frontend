import os
import io
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")

# Candidate column names
DATE_COLS = {"date", "order_date", "day", "datetime", "timestamp"}
PRODUCT_ID_COLS = {"product_id", "sku", "item_id", "id"}
PRODUCT_NAME_COLS = {"product_name", "name", "title"}
PRICE_COLS = {"price", "unit_price", "selling_price"}
UNITS_COLS = {"qty", "quantity", "units", "sold", "selling_number", "sales", "units_sold"}
RATING_COLS = {"rating", "stars", "review_rating"}

def _infer_col(df, candidates, numeric=False):
    for c in df.columns:
        if c.strip().lower() in candidates:
            if (not numeric) or pd.api.types.is_numeric_dtype(df[c]):
                return c
    # fallback: first numeric or first any
    if numeric:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    else:
        return df.columns[0] if len(df.columns) else None
    return None

def _prepare_sales(df, cols):
    d = df.copy()
    d[cols["date"]] = pd.to_datetime(d[cols["date"]], errors="coerce")
    d = d.dropna(subset=[cols["date"], cols["price"], cols["units"]])
    d[cols["price"]] = pd.to_numeric(d[cols["price"]], errors="coerce")
    d[cols["units"]] = pd.to_numeric(d[cols["units"]], errors="coerce")
    d = d.dropna(subset=[cols["price"], cols["units"]])
    # optional columns
    if cols.get("rating") and cols["rating"] in d.columns:
        d[cols["rating"]] = pd.to_numeric(d[cols["rating"]], errors="coerce")
    if cols.get("product_id") and cols["product_id"] in d.columns:
        d[cols["product_id"]] = d[cols["product_id"]].astype(str)
    if cols.get("product_name") and cols["product_name"] in d.columns:
        d[cols["product_name"]] = d[cols["product_name"]].astype(str)
    # revenue
    d["__revenue__"] = d[cols["price"]] * d[cols["units"]]
    d = d.sort_values(by=cols["date"]).reset_index(drop=True)
    return d

def _aggregate_daily(d, cols):
    daily = d.groupby(d[cols["date"]].dt.date).agg(
        revenue=("__revenue__", "sum"),
        units=(cols["units"], "sum"),
        orders=("__revenue__", "count")
    ).reset_index().rename(columns={cols["date"]]: "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["return_rev"] = daily["revenue"].pct_change().fillna(0.0)
    return daily

def _trend_slope(dates: pd.Series, values: pd.Series):
    x = dates.view('int64') // 10**9
    x = x.to_numpy().reshape(-1, 1)
    y = values.to_numpy().reshape(-1, 1)
    lr = LinearRegression().fit(x, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def _rolling_forecast(dates: pd.Series, values: pd.Series, steps=7):
    x = dates.view('int64') // 10**9
    x = x.to_numpy().reshape(-1, 1)
    y = values.to_numpy().reshape(-1, 1)
    lr = LinearRegression().fit(x, y)
    last = x[-1][0]
    step = int(np.median(np.diff(x.squeeze()))) if len(x) > 1 else 24*3600
    fut_dates = []
    preds = []
    for i in range(1, steps+1):
        ts = last + i*step
        fut_dates.append(pd.to_datetime(ts, unit='s').isoformat())
        preds.append(float(lr.predict([[ts]])[0][0]))
    return {"future_dates": fut_dates, "predictions": preds}

def _trending_products(d, cols):
    # recent 14 days split into last7 vs prev7
    if d.empty:
        return []
    max_date = d[cols["date"]].max().normalize()
    last7_start = max_date - pd.Timedelta(days=6)
    prev7_start = last7_start - pd.Timedelta(days=7)
    prev7_end = last7_start - pd.Timedelta(days=1)

    recent = d[d[cols["date"]].dt.date >= last7_start.date()]
    prev = d[(d[cols["date"]].dt.date >= prev7_start.date()) & (d[cols["date"]].dt.date <= prev7_end.date())]

    pid = cols.get("product_id") or cols.get("product_name")
    if not pid or pid not in d.columns:
        return []

    g_recent = recent.groupby(pid)[cols["units"]].sum().rename("u_recent")
    g_prev = prev.groupby(pid)[cols["units"]].sum().rename("u_prev")
    trend = pd.concat([g_recent, g_prev], axis=1).fillna(0.0)
    trend["growth"] = trend.apply(lambda r: (r["u_recent"] - r["u_prev"]) / (r["u_prev"] + 1e-9), axis=1)
    trend = trend.sort_values("growth", ascending=False).head(5).reset_index()
    # attach names and revenue
    if cols.get("product_name") and cols["product_name"] in d.columns:
        # ensure product_name available per product_id
        name_map = d.dropna(subset=[cols["product_name"]]).drop_duplicates(subset=[pid]).set_index(pid)[cols["product_name"]].to_dict()
        trend["product_name"] = trend[pid].map(name_map)
    # revenue recent
    rev_recent = recent.groupby(pid)["__revenue__"].sum().rename("revenue_recent")
    trend = trend.merge(rev_recent, on=pid, how="left")
    records = trend.to_dict(orient="records")
    # ensure strings for JSON
    for r in records:
        for k,v in list(r.items()):
            if isinstance(v, (np.floating, np.integer)):
                r[k] = float(v)
    return records

def _advisory(trend_slope, vol, anomaly_rate):
    tips = []
    if trend_slope > 0:
        tips.append("Overall revenue trend is UP — consider scaling inventory and marketing on top performers.")
    elif trend_slope < 0:
        tips.append("Overall revenue trend is DOWN — audit pricing, promotions, or product visibility.")
    else:
        tips.append("Revenue trend is flat — explore experiments on pricing and bundles.")
    if vol < 0.02:
        tips.append("Low day-to-day volatility — planning can be more aggressive.")
    elif vol < 0.05:
        tips.append("Moderate volatility — keep a buffer in operations and stock.")
    else:
        tips.append("High volatility — tighten forecasting windows; watch anomalies closely.")
    if anomaly_rate > 0.05:
        tips.append("Frequent anomalies — check for stockouts, flash sales, or tracking issues.")
    return tips

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/train", methods=["POST"])
def train():
    if 'file' not in request.files:
        return jsonify({"error": "Upload CSV in form field 'file'"}), 400
    f = request.files['file']
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    cols = {
        "date": request.form.get("date_col") or _infer_col(df, DATE_COLS),
        "product_id": request.form.get("product_id_col") or _infer_col(df, PRODUCT_ID_COLS),
        "product_name": request.form.get("product_name_col") or _infer_col(df, PRODUCT_NAME_COLS),
        "price": request.form.get("price_col") or _infer_col(df, PRICE_COLS, numeric=True),
        "units": request.form.get("units_col") or _infer_col(df, UNITS_COLS, numeric=True),
        "rating": request.form.get("rating_col") or _infer_col(df, RATING_COLS, numeric=True) if any(c in df.columns for c in RATING_COLS) else None
    }

    if not cols["date"] or not cols["price"] or not cols["units"]:
        return jsonify({"error": "Need date, price, and units columns (provide via form or ensure auto-detectable)."}), 400

    d = _prepare_sales(df, cols)
    daily = _aggregate_daily(d, cols)
    if len(daily) < 20:
        return jsonify({"error": "Need at least 20 daily rows after aggregation to train."}), 400

    # Isolation Forest on revenue returns
    X = daily[["return_rev"]].to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    iso = IsolationForest(random_state=42, contamination="auto").fit(Xs)

    slope, intercept = _trend_slope(daily["date"], daily["revenue"])
    vol = float(daily["return_rev"].rolling(20).std().dropna().mean() or daily["return_rev"].std())

    model = {
        "cols": cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "iso": iso,
        "trend_slope_ref": slope,
        "volatility_ref": vol,
        "ma_windows": [7, 28],
        "trained_at": datetime.utcnow().isoformat() + "Z"
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return jsonify({"message": "Model trained and saved.", "model_path": MODEL_PATH, "meta": {"trend_slope": slope, "volatility": vol, "cols": cols}})

@app.route("/analyze", methods=["POST"])
def analyze():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not found — train first."}), 400
    model = joblib.load(MODEL_PATH)
    cols = model["cols"]
    ma_windows = model.get("ma_windows", [7, 28])

    if 'file' not in request.files:
        return jsonify({"error": "Upload CSV in form field 'file'"}), 400
    f = request.files['file']
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    # If user uploads different schema, try inferring again
    if any(c not in df.columns for c in [cols["date"], cols["price"], cols["units"]]):
        cols = {
            "date": _infer_col(df, DATE_COLS),
            "product_id": _infer_col(df, PRODUCT_ID_COLS),
            "product_name": _infer_col(df, PRODUCT_NAME_COLS),
            "price": _infer_col(df, PRICE_COLS, numeric=True),
            "units": _infer_col(df, UNITS_COLS, numeric=True),
            "rating": _infer_col(df, RATING_COLS, numeric=True) if any(c in df.columns for c in RATING_COLS) else None
        }
        if not cols["date"] or not cols["price"] or not cols["units"]:
            return jsonify({"error": "Could not map date/price/units from CSV."}), 400

    d = _prepare_sales(df, cols)
    if d.empty:
        return jsonify({"error": "No valid rows after cleaning."}), 400

    daily = _aggregate_daily(d, cols)
    if len(daily) < 7:
        return jsonify({"error": "Need at least 7 daily rows for analysis."}), 400

    # moving averages
    for w in ma_windows:
        daily[f"ma_{w}"] = daily["revenue"].rolling(w).mean()

    # anomaly detection
    scaler_mean = np.array(model["scaler_mean"])
    scaler_scale = np.array(model["scaler_scale"])
    X = daily[["return_rev"]].to_numpy()
    Xs = (X - scaler_mean) / scaler_scale
    iso = model["iso"]
    pred = iso.predict(Xs)  # -1 anomaly
    score = iso.decision_function(Xs)
    daily["anomaly"] = (pred == -1).astype(int)
    daily["anom_score"] = score

    # trend/volatility fresh
    slope, intercept = _trend_slope(daily["date"], daily["revenue"])
    vol = float(daily["return_rev"].rolling(20).std().dropna().mean() or daily["return_rev"].std())
    anomaly_rate = float(daily["anomaly"].mean())
    forecast = _rolling_forecast(daily["date"], daily["revenue"], steps=7)
    advice = _advisory(slope, vol, anomaly_rate)

    # product leaderboard by revenue
    pid = cols.get("product_id") or cols.get("product_name")
    top_products = d.groupby(pid)["__revenue__"].sum().sort_values(ascending=False).head(10).reset_index()
    if cols.get("product_name") and cols["product_name"] in d.columns and pid != cols["product_name"]:
        name_map = d.dropna(subset=[cols["product_name"]]).drop_duplicates(subset=[pid]).set_index(pid)[cols["product_name"]].to_dict()
        top_products["product_name"] = top_products[pid].map(name_map)
    # ensure float
    top_products["revenue"] = top_products["__revenue__"].astype(float)
    top_products = top_products.drop(columns=["__revenue__"])

    # rating distribution (if available)
    rating_hist = None
    if cols.get("rating") and cols["rating"] in d.columns:
        valid = d[cols["rating"]].dropna()
        if not valid.empty:
            bins = [0,1,2,3,4,5]
            counts = pd.cut(valid, bins=bins, right=True, include_lowest=True).value_counts().sort_index()
            rating_hist = {"bins": [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)], "counts": counts.tolist()}

    # chart payloads
    chart = {
        "dates": daily["date"].dt.strftime("%Y-%m-%d").tolist(),
        "revenue": daily["revenue"].astype(float).tolist(),
        "units": daily["units"].astype(float).tolist(),
        "moving_averages": {f"ma_{w}": daily[f"ma_{w}"].fillna(None).tolist() for w in ma_windows},
        "anomalies": daily["anomaly"].astype(int).tolist(),
        "forecast": forecast
    }

    # trending products (growth last 7 vs prev 7)
    trending = _trending_products(d, cols)

    stats = {
        "total_revenue": float(d["__revenue__"].sum()),
        "total_units": float(d[cols["units"]].sum()),
        "avg_price": float(d[cols["price"]].mean()),
        "days": int(len(daily)),
        "volatility": vol,
        "trend_slope": slope,
        "anomaly_rate": anomaly_rate
    }

    return jsonify({
        "columns_used": cols,
        "basic_stats": stats,
        "advisory": advice,
        "chartData": chart,
        "topProducts": top_products.to_dict(orient="records"),
        "trendingProducts": trending,
        "ratingHistogram": rating_hist
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
