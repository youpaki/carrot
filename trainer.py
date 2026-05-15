"""
Carrot Bot — Multi-Job Training Pipeline.
Fetches ML data from PolyCore, runs parallel hyperparameter search,
auto-selects best model, reports results.
"""
import asyncio
import itertools
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

import config
from model import CarrotModel


def _train_job(X, y, params: dict) -> dict:
    """Train one model. Returns params + metrics."""
    try:
        model = CarrotModel()
        metrics = model.train(X, y, params)
        return {"params": params, **metrics, "ok": True}
    except Exception as e:
        return {"params": params, "ok": False, "error": str(e)}


class TrainingManager:
    def __init__(self):
        self.best_model: CarrotModel = None
        self.best_metrics: dict = {}
        self.running = False
        self.jobs_active = 0
        self.total_completed = 0
        self._executor = None

    async def fetch_data(self) -> tuple:
        """Fetch ML training dataset from PolyCore. Returns (X, y)."""
        url = f"{config.POLYCORE_URL}/tracker/export/ml/lite"
        params = {"only_resolved": True, "min_size_usdc": 1}
        if config.WHALE_FILTER:
            params["whale_address"] = config.WHALE_FILTER

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(url, params=params,
                                 headers={"X-API-Key": config.API_KEY})
            r.raise_for_status()
            data = r.json()

        rows = data.get("data", [])
        if not rows:
            return None, None

        df = pd.DataFrame(rows)
        feature_cols = [c for c in df.columns if c not in ("trade_id", "won", "pnl", "pnl_pct")]
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        X = df[feature_cols].values.astype(np.float32)
        y = df["won"].values.astype(int)
        return X, y

    def _param_grid(self):
        """Generate hyperparameter combinations to explore."""
        base = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.3],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 1.5, 2],
        }
        # Random sample from grid to avoid infinite combos
        all_keys = list(base.keys())
        combos = []
        for _ in range(config.TRAIN_MAX_ITER):
            combo = {}
            for k in random.sample(all_keys, min(len(all_keys), 6)):
                combo[k] = random.choice(base[k])
            combo["n_estimators"] = random.choice(base["n_estimators"])
            combo["max_depth"] = random.choice(base["max_depth"])
            combo["learning_rate"] = random.choice(base["learning_rate"])
            combos.append(combo)
        return combos

    async def run(self, model: CarrotModel, report_callback=None):
        """Main training loop. Runs until target accuracy or max iterations."""
        self.running = True
        self.total_completed = 0

        X, y = await self.fetch_data()
        if X is None or len(X) < 50:
            print(f"[Trainer] Not enough data: {len(X) if X is not None else 0} samples")
            self.running = False
            return

        print(f"[Trainer] Loaded {len(X)} samples, {X.shape[1]} features, "
              f"{int(y.sum())} wins, {int((1-y).sum())} losses")

        param_list = self._param_grid()
        print(f"[Trainer] Generated {len(param_list)} parameter combos, "
              f"max concurrent: {config.MAX_TRAIN_JOBS}")

        self._executor = ThreadPoolExecutor(max_workers=config.MAX_TRAIN_JOBS)
        futures = {}

        # Submit initial batch
        for params in param_list[:config.MAX_TRAIN_JOBS]:
            futures[self._executor.submit(_train_job, X, y, params)] = params

        next_idx = config.MAX_TRAIN_JOBS
        completed = 0

        while futures and self.running:
            done = []
            try:
                for fut in as_completed(futures.keys(), timeout=2):
                    done.append(fut)
            except TimeoutError:
                pass

            for fut in done:
                params = futures.pop(fut)
                result = fut.result()
                completed += 1
                self.total_completed = completed
                self.jobs_active = len(futures)

                if result.get("ok"):
                    acc = result.get("accuracy", 0)
                    brier = result.get("brier_score", 1)
                    print(f"[Trainer] #{completed}: acc={acc:.4f} brier={brier:.4f} "
                          f"n_est={params.get('n_estimators')} depth={params.get('max_depth')} "
                          f"lr={params.get('learning_rate')}")

                    if report_callback:
                        await report_callback(result)

                    # Track best
                    if not self.best_model or acc > self.best_metrics.get("accuracy", 0):
                        self.best_metrics = result
                        best = CarrotModel()
                        best.train(X, y, params)
                        self.best_model = best
                        best.save(config.MODEL_DIR / "best_model.pkl")
                        print(f"[Trainer] ★ New best: acc={acc:.4f}")

                        # Also update the caller's model
                        model.model = best.model
                        model.accuracy = best.accuracy
                        model.brier_score = best.brier_score
                        model.version = best.version

                    if acc >= config.TARGET_ACCURACY:
                        print(f"[Trainer] Target accuracy {config.TARGET_ACCURACY} reached! Stopping.")
                        self.running = False
                else:
                    print(f"[Trainer] #{completed}: ERROR {result.get('error', '')}")

                # Submit next if available
                if next_idx < len(param_list) and self.running:
                    params = param_list[next_idx]
                    futures[self._executor.submit(_train_job, X, y, params)] = params
                    next_idx += 1

            if not futures and next_idx >= len(param_list):
                break

        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

        self.jobs_active = 0
        self.running = False
        print(f"[Trainer] Done. {completed} jobs run. Best accuracy: "
              f"{self.best_metrics.get('accuracy', 0)}")

    def stop(self):
        self.running = False
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
