from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
from typing import List, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = FastAPI()

# ------------------
# Config
# ------------------

MODEL_PATH = "bart_mnli_model"
LABELS = ["real add", "dealer add"]
MAX_CHARS = 512
MAX_TEXTS = 50

BLACKLIST = [
    "total loss",
    "engine light",
    "engine light on",
    "rebuilt",
    "vag",
    "licensing omvic",
    "omvic",
    "email me",
    "email me at",
    "dealer"
]

# ------------------
# Load model once
# ------------------

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "zero-shot-classification",
    model=AutoModelForSequenceClassification.from_pretrained(MODEL_PATH),
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH),
    device=device
)

# ------------------
# Request models
# ------------------

class AdsObject(BaseModel):
    texts: List[str]

class PredictRequest(RootModel[Union[
    AdsObject,
    List[AdsObject]
]]):
    pass

# ------------------
# Utils
# ------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^a-z0-9\s]", " ", text)

def rule_based_check(text: str):
    cleaned = normalize_text(text)
    for k in BLACKLIST:
        if k in cleaned:
            return True, k
    return False, None

# ------------------
# Routes
# ------------------

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(payload: PredictRequest):

    data = payload.root

    if isinstance(data, list):
        texts = data[0].texts
    else:
        texts = data.texts

    if not texts:
        raise HTTPException(400, "texts is empty")

    if len(texts) > MAX_TEXTS:
        raise HTTPException(400, f"max {MAX_TEXTS} texts allowed")

    results = []
    batch = []
    mapping = []

    for i, text in enumerate(texts):
        text = text[:MAX_CHARS]
        flagged, keyword = rule_based_check(text)

        if flagged:
            results.append({
                "index": i,
                "label": "scam",
                "score": 1.0,
                "source": "rules",
                "matched_keyword": keyword
            })
        else:
            batch.append(text)
            mapping.append(i)

    if batch:
        outputs = classifier(
            batch,
            candidate_labels=LABELS,
            batch_size=8
        )

        for out, idx in zip(outputs, mapping):
            results.append({
                "index": idx,
                "label": "real" if "real" in out["labels"][0] else "dealer",
                "score": round(out["scores"][0], 4),
                "source": "model"
            })

    return {
        "total": len(results),
        "results": sorted(results, key=lambda x: x["index"])
    }
