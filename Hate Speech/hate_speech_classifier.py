from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Optional, Union, List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="HateSpeechDetector API",
    description="Local inference API for KoalaAI/HateSpeechDetector model",
    version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and tokenizers
MODEL_PATH = "D:/content_safeguard_model_api/Hate Speech/hub/models--KoalaAI--OffensiveSpeechDetector/snapshots/e25a05698ea3405bcd0ed309f3bde50f722ba8ec"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise


class InferenceRequest(BaseModel):
    inputs: Union[str, List[str]]


async def verify_token(authorization: Optional[str] = Header(None)):
    # Add actual token verification logic here
    return "Any token accepted for testing"


@app.post("/predict")
async def predict(request: InferenceRequest,
                  token: str = Header(None, alias="Authorization")):
    await verify_token(token)

    try:
        inputs = request.inputs if isinstance(request.inputs,
                                              list) else [request.inputs]
        results = []
        for text in inputs:
            # Tokenize
            encoded = tokenizer(text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512)

            # Get model outputs
            with torch.no_grad():
                outputs = model(**encoded)

            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Convert to list and get label
            probs = probabilities[0].tolist()
            label_id = torch.argmax(probabilities[0]).item()
            label = model.config.id2label[label_id]

            # Format result similar to Hugging Face API
            result = {"label": label, "score": probs[label_id]}
            results.append(result)

        # Return single result if input was a single string
        return results[0] if isinstance(request.inputs, str) else results

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error processing request: {str(e)}")


@app.get("/")
async def root():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
