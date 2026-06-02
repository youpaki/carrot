"""
Carrot Bot — Configuration.
Reads from .env file (managed by PolyCore dashboard).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def get(key, default=None):
    return os.environ.get(key, default)


POLYCORE_URL  = get("POLYCORE_URL", "https://poly.trenas.fr/polycore/api/v1")
WS_URL        = get("WS_URL", "wss://poly.trenas.fr/polycore/ws/events")
API_KEY       = get("POLYCORE_API_KEY", "")
BOT_ID        = get("BOT_ID", "")

DRY_RUN             = get("DRY_RUN", "true").lower() == "true"
INITIAL_BUDGET      = float(get("INITIAL_BUDGET", "200"))
COPY_RATIO          = float(get("COPY_RATIO", "0.002"))
MIN_STAKE_USDC      = float(get("MIN_STAKE_USDC", "1.0"))
MAX_CASH_PER_TRADE  = float(get("MAX_CASH_PER_TRADE", "50"))
CONFIDENCE_THRESHOLD = float(get("CONFIDENCE_THRESHOLD", "0.65"))
TARGET_ACCURACY     = float(get("TARGET_ACCURACY", "0.80"))
MAX_TRAIN_JOBS      = int(get("MAX_TRAIN_JOBS", "3"))
TRAIN_MAX_ITER      = int(get("TRAIN_MAX_ITER", "100"))
WHALE_FILTER        = get("WHALE_FILTER", "")
AUTO_TRAIN          = get("AUTO_TRAIN", "true").lower() == "true"
AUTO_RETRAIN        = get("AUTO_RETRAIN", "true").lower() == "true"
RETRAIN_MIN_NEW     = int(get("RETRAIN_MIN_NEW", "5000"))
WALLET_ID           = get("WALLET_ID", "")
MAX_PRICE           = float(get("MAX_PRICE", "0.85"))

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
