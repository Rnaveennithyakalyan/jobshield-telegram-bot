import requests
import time
import pickle
import os
from dotenv import load_dotenv

# ==============================
# LOAD ENV
# ==============================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not found in environment variables")

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

MODEL_PATH = "jobshield_model.pkl"
PIPELINE_DATA_PATH = "jobshield_pipeline.pkl"

# ==============================
# LOAD MODEL & VECTORIZER
# ==============================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PIPELINE_DATA_PATH, "rb") as f:
    data = pickle.load(f)
    vectorizer = data["vectorizer"]

print("✅ JobShield model & vectorizer loaded")

# ==============================
# TELEGRAM HELPERS
# ==============================
def get_updates(offset=None):
    url = BASE_URL + "/getUpdates"
    params = {"timeout": 100}
    if offset:
        params["offset"] = offset
    return requests.get(url, params=params).json()

def send_message(chat_id, text):
    url = BASE_URL + "/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload)

# ==============================
# ML PREDICTION
# ==============================
def analyze_job_description(text):
    X = vectorizer.transform([text])  # always 2D
    prediction = model.predict(X)[0]
    risk = model.predict_proba(X)[0][1] * 100

    verdict = "🚨 *FAKE JOB*" if prediction == 1 else "✅ *REAL JOB*"
    return verdict, round(risk, 2)

# ==============================
# MAIN LOOP
# ==============================
def main():
    offset = None
    print("🤖 JobShield Telegram Bot running permanently...")

    while True:
        updates = get_updates(offset)

        if not updates.get("ok"):
            time.sleep(2)
            continue

        for update in updates.get("result", []):
            offset = update["update_id"] + 1

            if "message" not in update:
                continue

            chat_id = update["message"]["chat"]["id"]
            text = update["message"].get("text", "").strip()

            if not text:
                continue

            if text.startswith("/start"):
                send_message(
                    chat_id,
                    "🛡️ *JobShield AI*\n\n"
                    "Send any job description and I will detect fake job risks."
                )
                continue

            verdict, risk = analyze_job_description(text)

            reply = (
                "🛡️ *JobShield AI Report*\n\n"
                f"📌 Verdict: {verdict}\n"
                f"⚠️ Risk Score: {risk}%"
            )

            send_message(chat_id, reply)

        time.sleep(1)

if __name__ == "__main__":
    main()
