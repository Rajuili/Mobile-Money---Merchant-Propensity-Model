from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

app = FastAPI(title="M-Pesa Lesotho Merchant Propensity API")

# Load synthetic for reference (optional)
try:
    df_ref = pd.read_csv("synthetic_mpesa_merchant.csv")
    BASE_RATE = df_ref['merchant_payer'].mean()
except:
    BASE_RATE = 0.215

class Customer(BaseModel):
    age: int = 39
    gender: str = "M"
    tenure: int = 6
    gsm: float = 41.0  # airtime+bundles
    fin_services: float = 0.0 # loans+od+savings
    inflow_outflow_ratio: float = 0.85
    p2p: float = 180.0
    bills: float = 0.0
    recency: int = 5
    product_mix: int = 6

@app.get("/")
def health():
    return {"status": "ok", "model": "mpesa-merchant-propensity-ls", "base_rate": BASE_RATE}

@app.post("/predict")
def predict(c: Customer):
    # Same rule-based model as before, no sklearn needed
    score = 0
    score += max(0, 15 - c.recency) * 2
    score += c.product_mix * 4
    score += np.log1p(c.fin_services) * 1.2
    score += np.log1p(c.p2p) * 0.3
    score += c.tenure * 0.8
    score += c.gsm * 0.01

    prob = 1 / (1 + np.exp(-(score - 30)/8))
    
    return {
        "msisdn": "2665XXXXXXX (anonymized)",
        "propensity_score": round(float(prob), 4),
        "will_pay_merchant_7d": bool(prob > 0.35),
        "threshold": 0.35,
        "inputs": c.dict()
    }