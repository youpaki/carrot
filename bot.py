#!/usr/bin/env python3
"""
Carrot Bot — Autonomous ML Copy-Trading Bot for PolyCore.
===========================================================
- Connects to PolyCore via WebSocket for real-time whale alerts
- Trains XGBoost models with multi-parameter search (ThreadPool)
- Copies whale trades when ML confidence exceeds threshold
- Hot-swaps model when training finds a better one
- Tracks portfolio locally — reports all trades to PolyCore
"""
import asyncio
import json
import os
import signal
import sys
import time
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
import websockets

import config
from model import CarrotModel, encode_event, features_array
from trainer import TrainingManager
from db import Persistence


class Portfolio:
    """Local portfolio: tracks positions, cash, realized PnL."""

    def __init__(self, initial_cash: float, persist: Persistence = None):
        self.initial_cash = initial_cash
        self.persist = persist
        self.positions: dict[str, dict] = {}
        self.trade_history: list[dict] = []
        self.total_trades = 0
        self.cash = initial_cash
        if persist:
            self._load()

    def _load(self):
        """Load state from SQLite."""
        data = self.persist.load_portfolio()
        if data["cash"] is not None:
            self.cash = data["cash"]
        if data["initial_cash"] is not None:
            self.initial_cash = data["initial_cash"]
        self.total_trades = data["total_trades"]
        # In live mode, don't load old paper positions — PolyCore is source of truth
        if not config.DRY_RUN:
            self.positions = {}
            self.trade_history = []
        else:
            self.positions = data["positions"]
            self.trade_history = data["trades"]
        if self.positions or self.trade_history:
            print(f"[DB] Loaded {len(self.positions)} positions, {len(self.trade_history)} trades, ${self.cash:.2f} cash")

    def _save(self):
        """Persist state to SQLite (called after every mutation)."""
        if self.persist:
            self.persist.save_portfolio(self)

    @property
    def total_invested(self) -> float:
        return sum(p.get("cost", 0) for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        realized = sum(t.get("pnl", 0) or 0 for t in self.trade_history if t.get("closed"))
        return realized

    @property
    def open_count(self) -> int:
        return len(self.positions)

    def open(self, market_id: str, shares: float, price: float, meta: dict = None) -> dict:
        """Open a position. shares = number of outcome tokens. cost = shares * price."""
        cost = shares * price
        pos = {
            "market_id": market_id,
            "shares": shares,
            "price": price,
            "cost": cost,
            "outcome": meta.get("outcome", "YES") if meta else "YES",
            "market_title": meta.get("market_title", "") if meta else "",
            "event_slug": meta.get("event_slug", "") if meta else "",
            "end_date": meta.get("end_date", "") if meta else "",
            "opened_at": datetime.now(timezone.utc),
        }
        # Merge with existing same-market BUY position
        for pid, p in list(self.positions.items()):
            if p["market_id"] == market_id:
                pos["shares"] += p["shares"]
                pos["cost"] += p["cost"]
                pos["price"] = pos["cost"] / pos["shares"] if pos["shares"] > 0 else price
                pos["opened_at"] = p["opened_at"]
                del self.positions[pid]
                break

        self.cash -= cost
        self.positions[str(uuid.uuid4())] = pos
        self.total_trades += 1
        self._save()
        return pos

    def close(self, pos_id: str, sell_price: float, shares: float = None) -> dict:
        """Close all or part of a position. Returns trade dict with PnL."""
        pos = self.positions.pop(pos_id, None)
        if not pos:
            return None
        s = shares if shares and shares < pos["shares"] else pos["shares"]
        fraction = s / pos["shares"]
        proceeds = s * sell_price
        cost_fraction = pos["cost"] * fraction
        pnl = round(proceeds - cost_fraction, 4)
        pnl_pct = round(pnl / cost_fraction, 4) if cost_fraction > 0 else 0
        self.cash += proceeds

        # If partially closed, put remaining back
        if s < pos["shares"]:
            pos["shares"] -= s
            pos["cost"] -= cost_fraction
            self.positions[str(uuid.uuid4())] = pos

        trade = {
            "market_id": pos["market_id"],
            "market_title": pos.get("market_title", ""),
            "event_slug": pos.get("event_slug", ""),
            "entry_price": pos["price"],
            "exit_price": sell_price,
            "shares": s,
            "cost": round(cost_fraction, 4),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "closed": True,
            "closed_at": datetime.now(timezone.utc),
        }
        self.trade_history.append(trade)
        self.total_trades += 1
        self._save()
        return trade


class CarrotBot:
    def __init__(self):
        self.model = CarrotModel()
        self._model_lock = threading.Lock()
        self.trainer = TrainingManager()
        self.persist = Persistence()
        self.portfolio = Portfolio(config.INITIAL_BUDGET, persist=self.persist)
        self.httpx_client = None
        self.whale_cache = {}
        self.uptime_start = None
        self.last_prediction = "—"
        self.total_predictions = 0
        self._last_resolved_count = 0
        self._resolved_new = 0
        self._events_received = 0
        self._recent_trades = set()  # dedup
        self._tasks = []
        self._live_positions = []
        self._live_trades = []

    # ── Model hot-swap (thread-safe) ───────────────────────────────────────

    def get_model(self):
        with self._model_lock:
            return self.model

    def set_model(self, new_model: CarrotModel):
        with self._model_lock:
            self.model = new_model

    # ── HTTP helpers ────────────────────────────────────────────────────────

    async def _api(self, method, path, body=None, params=None, timeout=15):
        url = f"{config.POLYCORE_URL}{path}"
        h = {"X-API-Key": config.API_KEY, "Content-Type": "application/json"}
        try:
            if self.httpx_client is None:
                self.httpx_client = httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=10, max_keepalive_connections=5))
            r = await self.httpx_client.request(method, url, json=body, params=params, headers=h, timeout=timeout)
            if r.status_code >= 400:
                detail = r.text[:500]
                print(f"[API] {r.status_code} {path}: {detail}")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            print(f"[API] HTTP {e.response.status_code} {path}: {e.response.text[:300]}")
            raise
        except Exception as e:
            print(f"[API] {type(e).__name__} {path}: {e}")
            raise

    async def _fix_missing_end_dates(self):
        """Re-fetch end_dates for positions that have empty end_dates."""
        missing = [(pid, pos) for pid, pos in self.portfolio.positions.items() if not pos.get("end_date")]
        if not missing:
            return
        print(f"[DB] Re-fetching end_dates for {len(missing)} positions...")
        for pid, pos in missing:
            try:
                ed = await self._fetch_market_end_date(
                    pos.get("market_id", ""),
                    event_slug=pos.get("event_slug", ""),
                    market_title=pos.get("market_title", ""),
                )
                if ed:
                    pos["end_date"] = ed
            except Exception:
                pass
        self.portfolio._save()

    async def _detect_wallet(self):
        """Fetch primary wallet from PolyCore for live trading."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{config.POLYCORE_URL}/bots/{config.BOT_ID}",
                    headers={"X-API-Key": config.API_KEY},
                )
                data = r.json()
                wallets = data.get("wallets", [])
                primary = next((w for w in wallets if w.get("role") == "primary"), wallets[0] if wallets else None)
                if primary:
                    config.WALLET_ID = primary["wallet_id"]
                    print(f"[Wallet] Using {primary.get('label', 'unknown')} ({config.WALLET_ID[:8]}...)")
                else:
                    print("[Wallet] No wallet found! Live orders will fail.")
        except Exception as e:
            print(f"[Wallet] Detection failed: {e}")

    async def _fetch_market_end_date(self, market_id: str, event_slug: str = "", market_title: str = "") -> str:
        """Fetch market end date from Polymarket Gamma API."""
        if not event_slug:
            return ""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"https://gamma-api.polymarket.com/events?slug={event_slug}")
                data = r.json()
                if isinstance(data, list) and data:
                    event = data[0]
                    # Try event-level endDate first
                    event_end = event.get("endDate", "")
                    # Find the specific market by matching title
                    markets = event.get("markets", [])
                    for m in markets:
                        q = (m.get("question") or "").lower().strip()
                        t = (market_title or "").lower().strip()
                        if q and t and (q.startswith(t[:40]) or t.startswith(q[:40])):
                            return m.get("endDate", "") or event_end
                    # Fallback: return event-level endDate
                    return event_end
        except Exception:
            pass
        return ""

    async def _report_trade(self, market_id: str, side: str, size: float, price: float, market_title: str, outcome: str, token_id: str = None, event_slug: str = "", status: str = "open", pnl: float = None):
        """Report a paper trade to PolyCore. Deduped per (market, side, timestamp window)."""
        if not config.DRY_RUN:
            return
        # Dedup: skip if same market+side+price in last 30s
        dedup_key = f"{market_id}:{side}:{round(size/price,2) if price>0 else size}"
        if dedup_key in self._recent_trades:
            return
        self._recent_trades.add(dedup_key)
        if len(self._recent_trades) > 200:
            self._recent_trades.clear()

        body = {
            "bot_id": config.BOT_ID,
            "market_id": market_id,
            "token_id": token_id or market_id,
            "question": market_title,
            "outcome": outcome,
            "side": side,
            "size": round(size / price, 2) if price > 0 else size,
            "price": price,
            "strategy": "carrot_ml",
            "meta": {"dry_run": True, "event_slug": event_slug},
            "status": status,
        }
        if pnl is not None:
            body["pnl"] = pnl
        try:
            await self._api("POST", "/trades", body)
        except Exception as e:
            print(f"[Trade] Report failed: {e}")

    async def _report_training(self, metrics):
        try:
            await self._api("POST", f"/bots/{config.BOT_ID}/train/result", {
                "run_id": str(uuid.uuid4()),
                "status": "completed",
                "model_version": self.get_model().version,
                "metrics": {**metrics},
            })
        except Exception as e:
            print(f"[API] Report training failed: {e}")

    async def _push_status(self):
        m = self.get_model()
        pf = self.portfolio
        try:
            await self._api("POST", f"/bots/{config.BOT_ID}/panel/status", {
                "model_version": m.version,
                "model_accuracy": f"{m.accuracy*100:.1f}%",
                "model_brier": f"{m.brier_score:.4f}",
                "last_prediction": self.last_prediction,
                "uptime": self._uptime_str(),
                "total_predictions": str(self.total_predictions),
                "training_jobs": f"{self.trainer.jobs_active} active / {self.trainer.total_completed} done",
                "resolved_new": str(self._resolved_new),
                "balance": f"${pf.cash + pf.total_invested:.2f}",
                "pnl": f"${pf.total_pnl:+.2f}",
                "positions": str(pf.open_count),
                "max_pos": str(max(1, int(pf.cash / max(config.MIN_STAKE_USDC, pf.cash * 0.01) / 2))),
                "trades": str(pf.total_trades),
                "cash": f"${pf.cash:.2f}",
                "invested": f"${pf.total_invested:.2f}",
                "mode": "DRY RUN" if config.DRY_RUN else "LIVE",
            })
        except Exception as e:
            pass

    # ── Panel ───────────────────────────────────────────────────────────────

    async def _declare_panel(self):
        await self._api("POST", f"/bots/{config.BOT_ID}/panel", {
            "sections": [
                {
                    "type": "fields",
                    "title": "Strategy",
                    "fields": [
                        {"key": "COPY_RATIO", "label": "Copy Ratio", "type": "number", "default": str(config.COPY_RATIO), "min": 0.0001, "max": 0.1, "step": 0.0005},
                        {"key": "MAX_CASH_PER_TRADE", "label": "Max Per Trade $", "type": "number", "default": str(config.MAX_CASH_PER_TRADE)},
                        {"key": "MIN_STAKE_USDC", "label": "Min Stake $", "type": "number", "default": str(config.MIN_STAKE_USDC)},
                        {"key": "CONFIDENCE_THRESHOLD", "label": "Confidence", "type": "number", "default": str(config.CONFIDENCE_THRESHOLD), "min": 0.5, "max": 0.95, "step": 0.01},
                        {"key": "TARGET_ACCURACY", "label": "Target Acc", "type": "number", "default": str(config.TARGET_ACCURACY), "min": 0.55, "max": 0.99, "step": 0.01},
                        {"key": "MAX_TRAIN_JOBS", "label": "Train Jobs", "type": "number", "default": str(config.MAX_TRAIN_JOBS), "min": 1, "max": 8, "step": 1},
                        {"key": "WHALE_FILTER", "label": "Whale Filter", "type": "text", "default": config.WHALE_FILTER},
                        {"key": "INITIAL_BUDGET", "label": "Budget $", "type": "number", "default": str(config.INITIAL_BUDGET)},
                        {"key": "RETRAIN_MIN_NEW", "label": "Retrain threshold", "type": "number", "default": str(config.RETRAIN_MIN_NEW), "step": 1000},
                    ]
                },
                {
                    "type": "buttons",
                    "title": "Commands",
                    "buttons": [
                        {"key": "train_now", "label": "Train", "style": "primary"},
                        {"key": "stop_training", "label": "Stop Train", "style": "danger"},
                        {"key": "reload_model", "label": "Reload Model", "style": "ghost"},
                        {"key": "reset_model", "label": "Reset", "style": "danger", "confirm": "Reset model?"},
                    ]
                },
                {
                    "type": "display",
                    "title": "Portfolio",
                    "items": [
                        {"key": "mode", "label": "Mode"},
                        {"key": "balance", "label": "Total Value"},
                        {"key": "cash", "label": "Cash"},
                        {"key": "invested", "label": "Invested"},
                        {"key": "pnl", "label": "Realized PnL"},
                        {"key": "positions", "label": "Open Positions"},
                        {"key": "max_pos", "label": "Max Positions"},
                        {"key": "trades", "label": "Total Trades"},
                        {"key": "model_version", "label": "Model"},
                        {"key": "model_accuracy", "label": "Accuracy"},
                        {"key": "model_brier", "label": "Brier"},
                        {"key": "last_prediction", "label": "Last Prediction"},
                        {"key": "uptime", "label": "Uptime"},
                        {"key": "total_predictions", "label": "Predictions"},
                        {"key": "training_jobs", "label": "Training"},
                        {"key": "resolved_new", "label": "New Resolved"},
                    ]
                },
            ]
        })
        print("[Panel] Registered")

    # ── Trading ──────────────────────────────────────────────────────────────

    async def _on_whale_trade(self, data: dict):
        model = self.get_model()
        if not model.model:
            return  # no trained model yet

        features = encode_event(data, self.whale_cache)
        X = features_array(features)
        prob = model.predict_proba(X)

        self._events_received += 1

        trade = data.get("trade", {})
        side = trade.get("side", "BUY").upper()
        market_id = trade.get("market_id", "")
        token_id = trade.get("token_id", "")
        event_slug = trade.get("event_slug", "") or trade.get("market_slug", "")
        market_title = trade.get("market_title", market_id[:40])
        price = float(trade.get("price", 0) or 0)

        # Skip 5-minute / 15-minute markets — resolve too fast to copy
        tl = market_title.lower()
        if any(x in tl for x in ["up or down", "5min", "5 min", "15min", "15 min", "minute"]):
            return

        if price <= 0 or price > 1:
            return
        outcome = trade.get("outcome", "YES")

        self.last_prediction = f"{side} @ {prob*100:.0f}% | {market_title[:40]}"
        self.total_predictions += 1

        if prob < config.CONFIDENCE_THRESHOLD:
            return

        # Dynamic position sizing: copy a fraction of what the whale put in
        whale_trade_size = float(data.get("trade", {}).get("size_usdc", 0) or 0)
        size_usdc = round(whale_trade_size * config.COPY_RATIO, 2)

        if self._events_received % 20 == 1:
            print(f"[Debug] {self._events_received} events, prob={prob*100:.0f}% side={side} whale=${whale_trade_size} copy=${size_usdc} cash=${self.portfolio.cash:.2f} open={self.portfolio.open_count}")
        if size_usdc < config.MIN_STAKE_USDC:
            if size_usdc > 0.1:
                pass  # too small — skip silently
            return
        size_usdc = min(size_usdc, config.MAX_CASH_PER_TRADE)

        # Dynamic max positions: more cash = more positions allowed
        avg_pos_size = max(config.MIN_STAKE_USDC, min(size_usdc, self.portfolio.cash * 0.05))
        max_positions = max(1, int(self.portfolio.cash / (avg_pos_size * 2)))
        size_usdc = round(size_usdc, 2)

        if side == "BUY":
            if self.portfolio.open_count >= max_positions:
                return
            if size_usdc > self.portfolio.cash:
                return

            shares = size_usdc / price  # convert USDC to outcome tokens
            if shares >= 15:
                order_type = "GTC"  # limit order — min 15 shares
            elif size_usdc >= 1:
                order_type = "FOK"  # market order — min $1 value
            else:
                return  # can't meet either minimum
            # CLOB requires: taker amount ≤4 decimals, maker amount ≤2 decimals
            shares = round(shares, 4)
            maker_usdc = round(shares * price, 2)
            if maker_usdc < 1:
                return  # below minimum
            print(f"[Trade] BUY ${maker_usdc:.2f} ({shares:.4f} sh @ {price:.4f}) {order_type} | {market_title[:50]} | prob={prob*100:.0f}%")

            # Place order FIRST (speed critical for short-lived markets)
            if not config.DRY_RUN:
                try:
                    await self._api("POST", "/clob/orders", {
                        "bot_id": config.BOT_ID,
                        "wallet_id": config.WALLET_ID,
                        "token_id": trade.get("token_id", market_id),
                        "side": "BUY", "size": shares, "price": price,
                        "order_type": order_type, "market_id": market_id,
                        "outcome": outcome, "strategy": "carrot_ml",
                    }, timeout=30)
                except Exception as e:
                    print(f"[Trade] Order fail: {e}")
                    return  # don't open position if order failed

            self.portfolio.open(market_id=market_id, shares=shares, price=price,
                               meta={"outcome": outcome, "market_title": market_title, "event_slug": event_slug, "end_date": ""})
            await self._report_trade(market_id, "BUY", size_usdc, price, market_title, outcome, token_id=token_id, event_slug=event_slug, status="open")

        else:  # SELL — only close existing positions in same market
            my_positions = [(pid, p) for pid, p in self.portfolio.positions.items()
                           if p["market_id"] == market_id]
            if not my_positions:
                return

            pid, pos = my_positions[0]
            sell_shares = min(size_usdc / price, pos["shares"]) if price > 0 else pos["shares"]

            # Determine order type based on share count
            if sell_shares >= 15 and pos["shares"] >= 15:
                order_type = "GTC"  # limit order — min 15 shares
            elif pos["shares"] < 15 and pos["shares"] * price >= 1:
                sell_shares = pos["shares"]  # close entire small position as market order
                order_type = "FOK"  # market order — min $1 value
            else:
                return  # can't meet either minimum

            sell_shares = round(sell_shares, 4)
            sell_usdc = round(sell_shares * price, 2)

            print(f"[Trade] SELL ${sell_usdc:.2f} ({sell_shares:.1f} sh @ {price:.4f}) {order_type} | {market_title[:50]} | prob={prob*100:.0f}%")

            result = self.portfolio.close(pid, price, sell_shares)
            if result:
                await self._report_trade(market_id, "SELL", sell_usdc, price, market_title, outcome, token_id=token_id, event_slug=event_slug, status="closed", pnl=result["pnl"])

            if not config.DRY_RUN:
                try:
                    await self._api("POST", "/clob/orders", {
                        "bot_id": config.BOT_ID,
                        "wallet_id": config.WALLET_ID,
                        "token_id": trade.get("token_id", market_id),
                        "side": "SELL", "size": sell_shares, "price": price,
                        "order_type": order_type, "market_id": market_id,
                        "outcome": outcome, "strategy": "carrot_ml",
                    })
                except Exception as e:
                    print(f"[Trade] Order fail: {e}")

    # ── Uptime ──────────────────────────────────────────────────────────────

    def _uptime_str(self):
        if not self.uptime_start:
            return "—"
        s = int((datetime.now(timezone.utc) - self.uptime_start).total_seconds())
        h, m = divmod(s, 3600)
        m, s = divmod(m, 60)
        return f"{h}h {m}m" if h else f"{m}m {s}s"

    # ── Retrain watcher ────────────────────────────────────────────────────

    async def _retrain_watcher(self):
        """Check resolved trade count every 5 min, trigger train if threshold met."""
        await asyncio.sleep(120)  # settle after startup
        while True:
            try:
                resp = await self._api("GET", "/tracker/export/manifest")
                resolved = int(resp.get("summary", {}).get("resolved_rows", 0) or 0)
                if self._last_resolved_count == 0:
                    self._last_resolved_count = resolved
                else:
                    self._resolved_new = resolved - self._last_resolved_count
                    # Force-stop stuck training (running >15min with 0 jobs)
                    if self.trainer.running and self.trainer.total_completed == 0 and self.trainer.started_at:
                        stuck_for = (datetime.now(timezone.utc) - self.trainer.started_at).total_seconds()
                        if stuck_for > 900:
                            print(f"[Retrain] Force-stopping stuck training ({stuck_for:.0f}s, 0 jobs)")
                            self.trainer.stop()
                            await asyncio.sleep(2)
                    if self._resolved_new >= config.RETRAIN_MIN_NEW and not self.trainer.running:
                        print(f"[Retrain] {self._resolved_new} new resolved trades (≥{config.RETRAIN_MIN_NEW}) → triggering")
                        self._last_resolved_count = resolved
                        self._resolved_new = 0
                        await self._start_training()
            except Exception as e:
                pass
            await asyncio.sleep(300)

    # ── Stale position cleaner ─────────────────────────────────────────────

    async def _try_fill(self, token_id: str, side: str, shares: float, price: float, market_title: str, timeout_sec: int = 120) -> dict:
        """Try to fill an order against the live orderbook. Retries every 1s until filled or timeout."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                fill = await self._api("POST", "/clob/simulate-fill", {
                    "token_id": token_id, "side": side, "size": shares, "price": price,
                }, timeout=8)
                if fill.get("filled"):
                    return fill
                if fill.get("reason") == "Orderbook unavailable":
                    pass  # keep retrying
            except Exception:
                pass
            await asyncio.sleep(1)
        return {"filled": False, "reason": f"Not filled after {timeout_sec}s"}

    # ── Stale position cleaner (disabled — positions close only via whale SELL) ──

    # ── WebSocket ───────────────────────────────────────────────────────────

    async def _ws_loop(self):
        url = f"{config.WS_URL}/{config.API_KEY}"
        while True:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=60, open_timeout=30, close_timeout=10) as ws:
                    print("[WS] Connected")
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        mtype = msg.get("type", "")
                        if mtype == "whale_trade":
                            await self._on_whale_trade(msg.get("data", msg))
                        elif mtype == "panel_action":
                            action = msg.get("data", {}).get("action", "")
                            await self._handle_action(action)
                        elif mtype == "heartbeat":
                            await ws.send(json.dumps({"type": "heartbeat"}))
                        elif mtype == "train_command":
                            await self._start_training()
            except Exception as e:
                print(f"[WS] {e}")
                await asyncio.sleep(5)

    # ── Actions ─────────────────────────────────────────────────────────────

    async def _handle_action(self, action: str):
        print(f"[Action] {action}")
        if action == "train_now":
            await self._start_training()
        elif action == "stop_training":
            self.trainer.stop()
        elif action == "reload_model":
            m = CarrotModel()
            if m.load(config.MODEL_DIR / "best_model.pkl"):
                self.set_model(m)
                print(f"[Model] Reloaded: {m.version}")
        elif action == "reset_model":
            self.set_model(CarrotModel())
            f = config.MODEL_DIR / "best_model.pkl"
            if f.exists():
                f.unlink()
            print("[Model] Reset")

    async def _start_training(self):
        if self.trainer.running:
            print("[Trainer] Already running")
            return
        print("[Trainer] Starting...")

        async def _do_train():
            m = self.get_model()
            await self.trainer.run(m, report_callback=self._report_training)
            # After training completes, update bot's model to the best found
            if self.trainer.best_model:
                self.set_model(self.trainer.best_model)
                print(f"[Trainer] Hot-swapped model: acc={self.trainer.best_metrics.get('accuracy',0):.4f}")

        self._tasks.append(asyncio.create_task(_do_train()))

    # ── Heartbeat ──────────────────────────────────────────────────────────

    async def _heartbeat_loop(self):
        while True:
            try:
                await self._api("POST", f"/bots/{config.BOT_ID}/heartbeat")
            except Exception:
                pass
            await asyncio.sleep(30)

    async def _status_loop(self):
        await asyncio.sleep(5)
        while True:
            try:
                await self._push_status()
            except Exception:
                pass
            # In live mode, sync real wallet balance periodically
            if not config.DRY_RUN:
                try:
                    await self._sync_live_wallet()
                except Exception:
                    pass
            await asyncio.sleep(30)

    async def _sync_live_wallet(self):
        """Fetch real USDC balance, positions, and trades from PolyCore."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                h = {"X-API-Key": config.API_KEY}
                base = config.POLYCORE_URL

                # Balance (fast endpoint)
                r = await client.get(f"{base}/clob/balance/{config.BOT_ID}", headers=h)
                r.raise_for_status()
                balance_raw = int(r.json().get("balance", {}).get("balance", 0))
                self.portfolio.cash = round(balance_raw / 1_000_000, 4)
                self.portfolio.initial_cash = self.portfolio.cash
        except Exception:
            pass
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                h = {"X-API-Key": config.API_KEY}
                base = config.POLYCORE_URL
                # Positions
                r = await client.get(f"{base}/clob/positions/{config.BOT_ID}", headers=h)
                r.raise_for_status()
                self._live_positions = r.json().get("positions", [])
                # Sync live positions into portfolio for SELL capability
                self.portfolio.positions = {}
                for i, lp in enumerate(self._live_positions):
                    market_id = lp.get("market", "")
                    if not market_id:
                        continue
                    shares = float(lp.get("size", 0))
                    avg_price = float(lp.get("avg_price", 0))
                    self.portfolio.positions[f"live_{i}"] = {
                        "market_id": market_id,
                        "shares": shares,
                        "price": avg_price,
                        "cost": shares * avg_price,
                        "outcome": lp.get("outcome", "YES"),
                        "market_title": lp.get("question", market_id[:40]),
                        "event_slug": "",
                        "end_date": "",
                        "opened_at": datetime.now(timezone.utc),
                    }
                self.portfolio._save()
                print(f"[Sync] {len(self._live_positions)} live positions, ${self.portfolio.cash:.2f} USDC")
        except Exception:
            pass
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                h = {"X-API-Key": config.API_KEY}
                base = config.POLYCORE_URL
                # Trades
                r = await client.get(f"{base}/clob/trades/{config.BOT_ID}", headers=h, params={"limit": 200})
                r.raise_for_status()
                self._live_trades = r.json().get("trades", [])
        except Exception:
            pass

    # ── State file dump (for dashboard) ──────────────────────────────────────

    def _dump_state(self):
        pf = self.portfolio
        m = self.get_model()

        # In live mode, use live data from PolyCore cache
        if not config.DRY_RUN and hasattr(self, '_live_positions'):
            positions = {}
            for i, p in enumerate(self._live_positions):
                positions[f"live_{i}"] = {
                    "market_id": p.get("market", ""),
                    "outcome": p.get("outcome", ""),
                    "shares": float(p.get("size", 0)),
                    "price": float(p.get("avg_price", 0)),
                    "cost": float(p.get("size", 0)) * float(p.get("avg_price", 0)),
                    "market_title": p.get("question", p.get("market", "")),
                }
            trade_history = []
            for t in self._live_trades:
                size = float(t.get("size", 0))
                price = float(t.get("price", 0))
                trade_history.append({
                    "market_id": t.get("market", ""),
                    "market_title": t.get("question", t.get("market", "")[:40]),
                    "entry_price": price,
                    "exit_price": price,
                    "shares": size,
                    "cost": size * price,
                    "pnl": 0,
                    "pnl_pct": 0,
                    "closed": t.get("status") == "SETTLED",
                    "closed_at": t.get("match_time", ""),
                    "side": t.get("side", ""),
                })
            cash = round(pf.cash, 4)
            total_invested = round(sum(p.get("size", 0) * p.get("avg_price", 0) for p in positions), 4)
            total_pnl = round(sum(t.get("pnl", 0) or 0 for t in trade_history if t.get("status") == "SETTLED"), 4)
            total_trades = len(trade_history)
        else:
            positions = {pid: {k: (str(v) if isinstance(v, datetime) else round(v, 6) if isinstance(v, float) else v) for k, v in p.items()} for pid, p in pf.positions.items()}
            trade_history = [{k: (str(v) if isinstance(v, datetime) else round(v, 6) if isinstance(v, float) else v) for k, v in t.items()} for t in pf.trade_history[-200:]]
            cash = round(pf.cash, 4)
            total_invested = round(pf.total_invested, 4)
            total_pnl = round(pf.total_pnl, 4)
            total_trades = pf.total_trades

        state = {
            "portfolio": {
                "cash": cash,
                "initial_cash": pf.initial_cash,
                "total_invested": total_invested,
                "total_pnl": total_pnl,
                "open_count": len(positions),
                "total_trades": total_trades,
                "positions": positions,
                "trade_history": trade_history,
            },
            "model": {
                "version": m.version,
                "accuracy": round(m.accuracy, 4) if m.accuracy else None,
                "brier_score": round(m.brier_score, 4) if m.brier_score else None,
                "has_model": m.model is not None,
            },
            "training": {
                "running": self.trainer.running,
                "jobs_active": self.trainer.jobs_active,
                "total_completed": self.trainer.total_completed,
                "best_accuracy": round(self.trainer.best_metrics.get("accuracy", 0), 4) if self.trainer.best_metrics else None,
                "best_brier": round(self.trainer.best_metrics.get("brier", 0), 4) if self.trainer.best_metrics else None,
            },
            "config": {
                "DRY_RUN": config.DRY_RUN,
                "INITIAL_BUDGET": config.INITIAL_BUDGET,
                "COPY_RATIO": config.COPY_RATIO,
                "MAX_CASH_PER_TRADE": config.MAX_CASH_PER_TRADE,
                "MIN_STAKE_USDC": config.MIN_STAKE_USDC,
                "CONFIDENCE_THRESHOLD": config.CONFIDENCE_THRESHOLD,
                "TARGET_ACCURACY": config.TARGET_ACCURACY,
                "MAX_TRAIN_JOBS": config.MAX_TRAIN_JOBS,
                "TRAIN_MAX_ITER": config.TRAIN_MAX_ITER,
                "WHALE_FILTER": config.WHALE_FILTER,
                "AUTO_TRAIN": config.AUTO_TRAIN,
                "AUTO_RETRAIN": config.AUTO_RETRAIN,
                "RETRAIN_MIN_NEW": config.RETRAIN_MIN_NEW,
            },
            "wallet": {
                "wallet_id": config.WALLET_ID,
                "balance_usdc": round(pf.cash, 4),
                "address": "got.more.money" if config.WALLET_ID else "",
            },
            "status": {
                "uptime": self._uptime_str(),
                "last_prediction": self.last_prediction,
                "total_predictions": self.total_predictions,
                "events_received": self._events_received,
                "resolved_new": self._resolved_new,
                "mode": "DRY RUN" if config.DRY_RUN else "LIVE",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            fpath = Path("/var/www/html/carrot") / "state.json"
            with open(fpath, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.chmod(fpath, 0o644)
        except Exception:
            pass
        try:
            epath = Path("/var/www/html/carrot") / "env.json"
            env_vars = {k: v for k, v in vars(config).items()
                        if not k.startswith("_") and k.isupper() and isinstance(v, (str, int, float, bool))}
            with open(epath, "w") as f:
                json.dump(env_vars, f, indent=2, default=str)
            os.chmod(epath, 0o644)
        except Exception:
            pass

    def _dump_history(self):
        try:
            pf = self.portfolio
            m = self.get_model()
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "cash": round(pf.cash, 4),
                "invested": round(pf.total_invested, 4),
                "value": round(pf.cash + pf.total_invested, 4),
                "pnl": round(pf.total_pnl, 4),
                "positions": pf.open_count,
                "trades": pf.total_trades,
                "accuracy": round(m.accuracy, 4) if m.accuracy else None,
                "model_version": m.version,
                "training": self.trainer.running,
            }
            self.persist.save_history_snapshot(entry)
            self.persist.trim_history(500)
        except Exception:
            pass

    async def _state_dump_loop(self):
        await asyncio.sleep(10)
        dump_count = 0
        while True:
            try:
                self._dump_state()
            except Exception as e:
                print(f"[Dump] State dump failed: {type(e).__name__}: {e}")
            try:
                await self._check_command_file()
            except Exception:
                pass
            dump_count += 1
            if dump_count % 4 == 0:
                try:
                    self._dump_history()
                except Exception as e:
                    print(f"[Dump] History dump failed: {type(e).__name__}: {e}")
            await asyncio.sleep(15)

    # ── Command file watcher (for dashboard control) ─────────────────────────

    async def _check_command_file(self):
        cmd_path = Path("/tmp/carrot_command.json")
        try:
            if cmd_path.exists():
                content = cmd_path.read_text()
                data = json.loads(content)
                action = data.get("command", "")
                if action:
                    print(f"[Dashboard] Command: {action}")
                    await self._handle_action(action)
                cmd_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[Dashboard] Command error: {e}")
            try:
                cmd_path.unlink(missing_ok=True)
            except Exception:
                pass

    @property
    def _state(self) -> dict:
        pf = self.portfolio
        m = self.get_model()
        sa = self.trainer.started_at
        return {
            "portfolio": {
                "cash": round(pf.cash, 4),
                "initial_cash": pf.initial_cash,
                "total_invested": round(pf.total_invested, 4),
                "total_pnl": round(pf.total_pnl, 4),
                "open_count": pf.open_count,
                "total_trades": pf.total_trades,
                "positions": {
                    pid: {k: (str(v) if isinstance(v, datetime) else round(v, 6) if isinstance(v, float) else v) for k, v in p.items()}
                    for pid, p in pf.positions.items()
                },
                "trade_history": [
                    {k: (str(v) if isinstance(v, datetime) else round(v, 6) if isinstance(v, float) else v) for k, v in t.items()}
                    for t in pf.trade_history[-200:]
                ],
            },
            "model": {
                "version": m.version,
                "accuracy": round(m.accuracy, 4) if m.accuracy else None,
                "brier_score": round(m.brier_score, 4) if m.brier_score else None,
                "has_model": m.model is not None,
            },
            "training": {
                "running": self.trainer.running,
                "jobs_active": self.trainer.jobs_active,
                "total_completed": self.trainer.total_completed,
                "best_accuracy": round(self.trainer.best_metrics.get("accuracy", 0), 4) if self.trainer.best_metrics else None,
                "best_brier": round(self.trainer.best_metrics.get("brier", 0), 4) if self.trainer.best_metrics else None,
                "started_at": str(sa) if sa else None,
                "elapsed": str(datetime.now(timezone.utc) - sa).split('.')[0] if sa and self.trainer.running else None,
            },
            "config": {
                "DRY_RUN": config.DRY_RUN,
                "INITIAL_BUDGET": config.INITIAL_BUDGET,
                "COPY_RATIO": config.COPY_RATIO,
                "MAX_CASH_PER_TRADE": config.MAX_CASH_PER_TRADE,
                "MIN_STAKE_USDC": config.MIN_STAKE_USDC,
                "CONFIDENCE_THRESHOLD": config.CONFIDENCE_THRESHOLD,
                "TARGET_ACCURACY": config.TARGET_ACCURACY,
                "MAX_TRAIN_JOBS": config.MAX_TRAIN_JOBS,
                "TRAIN_MAX_ITER": config.TRAIN_MAX_ITER,
                "WHALE_FILTER": config.WHALE_FILTER,
                "AUTO_TRAIN": config.AUTO_TRAIN,
                "AUTO_RETRAIN": config.AUTO_RETRAIN,
                "RETRAIN_MIN_NEW": config.RETRAIN_MIN_NEW,
            },
            "status": {
                "uptime": self._uptime_str(),
                "last_prediction": self.last_prediction,
                "total_predictions": self.total_predictions,
                "events_received": self._events_received,
                "resolved_new": self._resolved_new,
                "mode": "DRY RUN" if config.DRY_RUN else "LIVE",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Main ────────────────────────────────────────────────────────────────

    async def run(self):
        print(f"[Carrot] DRY_RUN={config.DRY_RUN} | Budget=${self.portfolio.initial_cash} | BOT={config.BOT_ID[:8]}...")
        self.uptime_start = datetime.now(timezone.utc)

        self.get_model().load(config.MODEL_DIR / "best_model.pkl")

        # Re-fetch end_dates for positions missing them
        await self._fix_missing_end_dates()

        try:
            await self._declare_panel()
        except Exception as e:
            print(f"[Panel] Declare failed: {e}")

        # Auto-detect wallet from PolyCore
        if not config.WALLET_ID and not config.DRY_RUN:
            await self._detect_wallet()

        if config.AUTO_TRAIN:
            print("[Trainer] Auto-train enabled")
            await self._start_training()

        if config.AUTO_RETRAIN:
            print(f"[Retrain] Watcher enabled — threshold {config.RETRAIN_MIN_NEW} new resolved trades")
            self._tasks.append(asyncio.create_task(self._retrain_watcher()))

        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._status_loop()))
        self._tasks.append(asyncio.create_task(self._state_dump_loop()))
        await self._ws_loop()


def main():
    bot = CarrotBot()

    def _shutdown():
        print("\n[Carrot] Shutting down...")
        bot.trainer.stop()
        try:
            bot.portfolio._save()
            print("[Carrot] Portfolio saved to DB")
        except Exception as e:
            print(f"[Carrot] DB save on shutdown failed: {e}")
        for t in bot._tasks:
            t.cancel()

    signal.signal(signal.SIGINT, lambda s, f: _shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: _shutdown())
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
