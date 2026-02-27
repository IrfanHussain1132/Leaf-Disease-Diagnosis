from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import requests
import os
import json
from model import predict

app = FastAPI()

# ==============================
# CORS (Frontend Connection)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Load API Key
# ==============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==============================
# Root Endpoint
# ==============================
@app.get("/")
def home():
    return {"message": "AgriAI Backend Running 🚀"}


# ==============================
# Prediction Endpoint
# ==============================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 1️⃣ Model Prediction
        result = predict(image)

        # 2️⃣ Confidence Threshold
        if result["confidence"] < 75:
            return {
                "disease": "Uncertain Prediction",
                "confidence": result["confidence"]
            }

        # 3️⃣ Get Cure Info from Groq
        cure_data = get_cure_from_groq(result["disease"])

        return {
            "disease": result["disease"],
            "confidence": result["confidence"],
            **cure_data
        }

    except Exception as e:
        return {"error": str(e)}


# ==============================
# Groq API Function
# ==============================
# ==============================
# Groq API Function (Safe Version)
# ==============================
def get_cure_from_groq(disease: str):
    if not GROQ_API_KEY:
        print("❌ GROQ API key not found")
        return fallback_response()

    prompt = f"""
You are an agricultural plant disease expert.

Provide clear, practical and farmer-friendly treatment details for {disease}.

Rules:
- Keep language simple and easy to understand.
- Give specific examples (real fungicide names if possible).
- Mention measurable dosage (ml/L or g/L).
- Avoid long explanations.
- Keep each field short and precise.
- Return ONLY valid JSON.

Return JSON in this exact format:

{{
    "traditional": {{
        "method": "",
        "frequency": "",
        "effect": ""
    }},
    "chemical": {{
        "name": "",
        "dosage": "",
        "interval": "",
        "timeline": ""
    }},
    "prevention": []
}}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )

        if response.status_code != 200:
            print("❌ Groq API Error:", response.text)
            return fallback_response()

        result = response.json()

        # Safe access
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not content:
            print("❌ Empty response from Groq")
            return fallback_response()

        # Try direct JSON parsing first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If model adds extra text, extract JSON safely
            start = content.find("{")
            end = content.rfind("}") + 1

            if start == -1 or end == -1:
                print("❌ JSON not found in response")
                return fallback_response()

            json_string = content[start:end]
            return json.loads(json_string)

    except requests.exceptions.RequestException as e:
        print("❌ Request failed:", str(e))
        return fallback_response()

    except Exception as e:
        print("❌ Unexpected error:", str(e))
        return fallback_response()

# ==============================
# Fallback (If Groq Fails)
# ==============================
def fallback_response():
    return {
        "traditional": {
            "method": "Apply recommended organic treatment.",
            "frequency": "Twice weekly.",
            "effect": "Visible improvement in 5-7 days."
        },
        "chemical": {
            "name": "Standard fungicide treatment.",
            "dosage": "Follow manufacturer instructions.",
            "interval": "7 day interval.",
            "timeline": "Stops spread within 48 hours."
        },
        "prevention": [
            "Maintain proper plant spacing.",
            "Remove infected leaves.",
            "Avoid overhead irrigation.",
            "Use resistant plant varieties."
        ]
    }