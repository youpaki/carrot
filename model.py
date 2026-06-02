"""
Carrot Bot — XGBoost ML Model.
Feature encoding, training, prediction, persistence.
"""
import pickle
import numpy as np
import xgboost as xgb
from datetime import datetime, timezone
from pathlib import Path

import config

FEATURE_NAMES = [
    "whale_trust_score",
    "whale_total_volume_usdc",
    "whale_total_trades",
    "whale_win_rate",
    "whale_avg_roi",
    "outcome_index",
    "side_encoded",
    "price",
    "size_usdc",
    "hour_of_day",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "price_deviation",
    "volume_ratio",
    "is_entry",
    "is_exit",
    "market_volume",
    "market_liquidity",
    "price_before",
    "price_impact",
    "days_to_resolution",
    "market_close_soon",
    "whale_age_months",
    "market_category_id",
]


def encode_event(evt: dict, whale_cache: dict = None) -> dict:
    """Encode a whale_trade WS event to feature dict."""
    d = evt.get("data", evt) if isinstance(evt, dict) else {}
    trade = d.get("trade", d)
    ts_str = trade.get("timestamp", "")
    ts = None
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        ts = datetime.now(timezone.utc)

    outcome = (trade.get("outcome", "") or "").lower()
    outcome_index = trade.get("outcome_index")
    if outcome_index is None:
        outcome_index = 0 if outcome in ("yes", "up") else 1

    whale_addr = d.get("whale_address", "")
    wc = (whale_cache or {}).get(whale_addr, {})
    whale_trust = float(wc.get("trust_score", 0.5) or 0.5)
    whale_volume = float(wc.get("total_volume_usdc", 0) or 0)
    whale_trades = float(wc.get("total_trades", 0) or 0)

    slug = trade.get("market_slug", "")
    market_category_id = _category_id(slug)

    # Compute days_to_resolution and market_close_soon from market_end_date
    market_end_date_str = trade.get("market_end_date")
    days_to_resolution = 30.0
    market_close_soon = 0
    if market_end_date_str:
        try:
            end_date = datetime.fromisoformat(market_end_date_str.replace("Z", "+00:00"))
            diff_days = (end_date - ts).total_seconds() / 86400
            days_to_resolution = max(0.0, round(diff_days, 1))
            market_close_soon = 1 if diff_days <= 7 else 0
        except Exception:
            pass

    # Compute whale_age_months from whale_cache
    whale_age_months = 6.0
    added_at_str = wc.get("added_at")
    if added_at_str:
        try:
            added = datetime.fromisoformat(added_at_str.replace("Z", "+00:00"))
            whale_age_months = max(0.0, round((ts - added).total_seconds() / (86400 * 30), 1))
        except Exception:
            pass

    return {
        "whale_trust_score": whale_trust,
        "whale_total_volume_usdc": whale_volume,
        "whale_total_trades": whale_trades,
        "whale_win_rate": float(wc.get("win_rate", 0.5) or 0.5),
        "whale_avg_roi": float(wc.get("avg_roi", 0) or 0),
        "outcome_index": outcome_index,
        "side_encoded": 1 if trade.get("side", "BUY") == "BUY" else 0,
        "price": float(trade.get("price", 0) or 0),
        "size_usdc": float(trade.get("size_usdc", 0) or 0),
        "hour_of_day": ts.hour,
        "day_of_week": ts.weekday(),
        "hour_sin": __import__("math").sin(2 * __import__("math").pi * ts.hour / 24),
        "hour_cos": __import__("math").cos(2 * __import__("math").pi * ts.hour / 24),
        "day_sin": __import__("math").sin(2 * __import__("math").pi * ts.weekday() / 7),
        "day_cos": __import__("math").cos(2 * __import__("math").pi * ts.weekday() / 7),
        "price_deviation": abs(float(trade.get("price", 0) or 0) - 0.5),
        "volume_ratio": min(float(trade.get("size_usdc", 0) or 0) / max(float(trade.get("market_volume", 1) or 1), 1), 1),
        "is_entry": 1 if trade.get("is_entry") else 0,
        "is_exit": 1 if trade.get("is_exit") else 0,
        "market_volume": float(trade.get("market_volume", 0) or 0),
        "market_liquidity": float(trade.get("market_liquidity", 0) or 0),
        "price_before": float(trade.get("price_before", 0) or 0),
        "price_impact": float(trade.get("price_impact", 0) or 0),
        "days_to_resolution": days_to_resolution,
        "market_close_soon": market_close_soon,
        "whale_age_months": whale_age_months,
        "market_category_id": market_category_id,
    }


def _category_id(slug: str) -> int:
    s = (slug or "").lower()
    if any(k in s for k in ["btc", "bitcoin", "eth", "ethereum", "crypto", "solana", "xrp", "updown", "up or down", "price of"]):
        return 0
    if any(k in s for k in ["sport", "nba", "nfl", "mlb", "soccer", "football", "basketball", "baseball", "hockey", "tennis", "ufc", "boxing", "league", "champion", "win the", "lose the"]):
        return 1
    if any(k in s for k in ["politic", "president", "congress", "election", "democrat", "republican", "vote", "war", "tariff", "inaugur", "bill"]):
        return 2
    return 3


def features_array(encoded: dict) -> np.ndarray:
    """Convert feature dict to numpy array in correct order."""
    return np.array([[encoded.get(n, 0) for n in FEATURE_NAMES]], dtype=np.float32)


class CarrotModel:
    def __init__(self):
        self.model: xgb.XGBClassifier = None
        self.version: str = "untrained"
        self.accuracy: float = 0.0
        self.brier_score: float = 1.0

    def train(self, X_train, y_train, X_test, y_test, params: dict = None) -> dict:
        """Train with given params. X/y already split chronologically. Returns metrics dict."""
        from sklearn.metrics import accuracy_score, brier_score_loss

        p = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        p.update(params or {})

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos > 0 and n_neg > 0:
            p["scale_pos_weight"] = n_neg / n_pos

        # Use pre-split data (time-based split from trainer)
        self.model = xgb.XGBClassifier(**p)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        brier = float(brier_score_loss(y_test, y_prob))
        logloss = brier

        # Baseline: accuracy if we always predicted the majority class
        n_pos_test = int(y_test.sum())
        n_neg_test = len(y_test) - n_pos_test
        baseline_acc = max(n_pos_test, n_neg_test) / len(y_test) if len(y_test) > 0 else 0.5
        edge = round(acc - baseline_acc, 4)

        self.accuracy = acc
        self.brier_score = brier
        self.version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        return {
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(logloss, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "edge": edge,
            "win_rate": round(acc, 4),
            "n_samples": int(len(y_test)),
            "n_samples_train": int(len(y_train)),
            "n_features": X_train.shape[1],
            "n_positives": int(n_pos_test),
            "n_negatives": int(n_neg_test),
            "scale_pos_weight": round(p.get("scale_pos_weight", 0), 4),
            "n_estimators": p["n_estimators"],
            "max_depth": p["max_depth"],
            "learning_rate": p["learning_rate"],
        }

    def predict_proba(self, X: np.ndarray) -> float:
        """Predict win probability for a single sample."""
        if self.model is None:
            return 0.5
        return float(self.model.predict_proba(X)[0, 1])

    def predict(self, feats: dict) -> tuple:
        """Convenience: encode + predict. Returns (prob, features_array)."""
        X = features_array(feats)
        return self.predict_proba(X), None

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "version": self.version,
                          "accuracy": self.accuracy, "brier_score": self.brier_score}, f)

    def load(self, path: Path):
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.version = data.get("version", "unknown")
        self.accuracy = data.get("accuracy", 0)
        self.brier_score = data.get("brier_score", 1)
        return True
