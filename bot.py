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


class Portfolio:
    """Local portfolio: tracks positions, cash, realized PnL."""

    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, dict] = {}
        self.trade_history: list[dict] = []
        self.total_trades = 0

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
        return trade


class CarrotBot:
    def __init__(self):
        self.model = CarrotModel()
        self._model_lock = threading.Lock()
        self.trainer = TrainingManager()
        self.portfolio = Portfolio(config.INITIAL_BUDGET)
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

    # ── Model hot-swap (thread-safe) ───────────────────────────────────────

    def get_model(self):
        with self._model_lock:
            return self.model

    def set_model(self, new_model: CarrotModel):
        with self._model_lock:
            self.model = new_model

    # ── HTTP helpers ────────────────────────────────────────────────────────

    async def _api(self, method, path, body=None, params=None, timeout=30):
        url = f"{config.POLYCORE_URL}{path}"
        h = {"X-API-Key": config.API_KEY, "Content-Type": "application/json"}
        if self.httpx_client is None:
            self.httpx_client = httpx.AsyncClient(timeout=timeout)
        r = await self.httpx_client.request(method, url, json=body, params=params, headers=h)
        r.raise_for_status()
        return r.json()

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
                "metrics": metrics,
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
        event_slug = trade.get("event_slug", "")
        market_title = trade.get("market_title", market_id[:40])
        price = float(trade.get("price", 0) or 0)
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
            print(f"[Debug] {self._events_received} events, prob={prob*100:.0f}% whale=${whale_trade_size} copy=${size_usdc}")
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
            # Try to fill against live orderbook for up to 2 minutes
            if config.DRY_RUN and token_id:
                fill = await self._try_fill(token_id, "BUY", shares, price, market_title, timeout_sec=120)
                if not fill.get("filled"):
                    print(f"[Trade] SKIP BUY (fill timeout): {market_title[:40]} | {fill.get('reason','')}")
                    return
                else:
                    print(f"[Trade] Fill OK at ask={fill.get('avg_fill_price')}")

            print(f"[Trade] BUY ${size_usdc:.2f} ({shares:.1f} sh @ {price:.4f}) | {market_title[:50]} | prob={prob*100:.0f}%")

            self.portfolio.open(market_id=market_id, shares=shares, price=price,
                               meta={"outcome": outcome, "market_title": market_title})
            await self._report_trade(market_id, "BUY", size_usdc, price, market_title, outcome, token_id=token_id, event_slug=event_slug, status="open")

            if not config.DRY_RUN:
                try:
                    await self._api("POST", "/clob/orders", {
                        "bot_id": config.BOT_ID,
                        "token_id": trade.get("token_id", market_id),
                        "side": "BUY", "size": shares, "price": price,
                        "order_type": "GTC", "market_id": market_id,
                        "outcome": outcome, "strategy": "carrot_ml",
                    })
                except Exception as e:
                    print(f"[Trade] Order fail: {e}")

        else:  # SELL — only close existing positions in same market
            my_positions = [(pid, p) for pid, p in self.portfolio.positions.items()
                           if p["market_id"] == market_id]
            if not my_positions:
                return

            pid, pos = my_positions[0]
            sell_shares = min(size_usdc / price, pos["shares"]) if price > 0 else pos["shares"]
            sell_usdc = sell_shares * price

            # Try to fill against live orderbook for up to 2 minutes
            if config.DRY_RUN and token_id:
                fill = await self._try_fill(token_id, "SELL", sell_shares, price, market_title, timeout_sec=120)
                if not fill.get("filled"):
                    print(f"[Trade] SKIP SELL (fill timeout): {market_title[:40]} | {fill.get('reason','')}")
                    return
                else:
                    print(f"[Trade] Fill OK at bid={fill.get('avg_fill_price')}")
                    if fill.get("avg_fill_price") and fill["avg_fill_price"] != price:
                        sell_shares = fill.get("filled_size", sell_shares)
                        sell_usdc = sell_shares * fill["avg_fill_price"]
                return

            print(f"[Trade] SELL ${sell_usdc:.2f} ({sell_shares:.1f} sh @ {price:.4f}) | {market_title[:50]} | prob={prob*100:.0f}%")

            result = self.portfolio.close(pid, price, sell_shares)
            if result:
                await self._report_trade(market_id, "SELL", sell_usdc, price, market_title, outcome, token_id=token_id, event_slug=event_slug, status="closed", pnl=result["pnl"])

            if not config.DRY_RUN:
                try:
                    await self._api("POST", "/clob/orders", {
                        "bot_id": config.BOT_ID,
                        "token_id": trade.get("token_id", market_id),
                        "side": "SELL", "size": close_size / price, "price": price,
                        "order_type": "GTC", "market_id": market_id,
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

    async def _stale_cleaner(self):
        """Auto-close positions older than 2 minutes that never had a real fill."""
        await asyncio.sleep(60)
        while True:
            try:
                now = datetime.now(timezone.utc)
                stale = [(pid, pos) for pid, pos in list(self.portfolio.positions.items())
                         if (now - pos["opened_at"]).total_seconds() > 120]
                for pid, pos in stale:
                    result = self.portfolio.close(pid, pos["price"], pos["shares"])
                    if result:
                        await self._report_trade(
                            pos["market_id"], "SELL",
                            pos["shares"] * pos["price"], pos["price"],
                            pos.get("market_title", ""),
                            pos.get("outcome", "YES"),
                            token_id=pos["market_id"],
                            status="closed", pnl=0)
                        print(f"[Stale] Closed {pos['market_id'][:20]}... after {(now - pos['opened_at']).total_seconds():.0f}s")
            except Exception as e:
                print(f"[Stale] Error: {e}")
            await asyncio.sleep(30)

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
            await asyncio.sleep(30)

    # ── Main ────────────────────────────────────────────────────────────────

    async def run(self):
        print(f"[Carrot] DRY_RUN={config.DRY_RUN} | Budget=${self.portfolio.initial_cash} | BOT={config.BOT_ID[:8]}...")
        self.uptime_start = datetime.now(timezone.utc)

        self.get_model().load(config.MODEL_DIR / "best_model.pkl")

        try:
            await self._declare_panel()
        except Exception as e:
            print(f"[Panel] Declare failed: {e}")

        if config.AUTO_TRAIN:
            print("[Trainer] Auto-train enabled")
            await self._start_training()

        if config.AUTO_RETRAIN:
            print(f"[Retrain] Watcher enabled — threshold {config.RETRAIN_MIN_NEW} new resolved trades")
            self._tasks.append(asyncio.create_task(self._retrain_watcher()))

        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._status_loop()))
        self._tasks.append(asyncio.create_task(self._stale_cleaner()))
        await self._ws_loop()


def main():
    bot = CarrotBot()

    def _shutdown():
        print("\n[Carrot] Shutting down...")
        bot.trainer.stop()
        for t in bot._tasks:
            t.cancel()

    signal.signal(signal.SIGINT, lambda s, f: _shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: _shutdown())
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
