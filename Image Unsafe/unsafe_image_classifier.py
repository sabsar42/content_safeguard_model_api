from fastapi import FastAPI, HTTPException, File, UploadFile
from transformers import AutoModelForImageClassification, AutoImageProcessor  # Changed this line
import torch
from PIL import Image
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ImageClassification API",
    description="Image Classification API for NSFW content detection",
    version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(
    r"D:\content_safeguard_model_api\Image Unsafe\models--MichalMlodawski--nsfw-image-detection-large\snapshots\631537d3bb871f59abb2769c1127359920a73640"
)

model = None
image_processor = None


def load_model():
    """Initialize the model and image processor"""
    global model, image_processor

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

        logger.info("Loading model and image processor...")
        image_processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
        model = AutoModelForImageClassification.from_pretrained(
            str(MODEL_PATH))
        model.eval()
        logger.info("Model and image processor loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model or image processor: {str(e)}")
        raise RuntimeError(
            f"Failed to load model and image processor: {str(e)}")


load_model()


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=
            f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    try:
        image = Image.open(file.file)

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            label_id = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][label_id].item()
            label = model.config.id2label[label_id]

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Error processing image: {str(e)}")
    finally:
        file.file.close()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model_loaded": model is not None and image_processor is not None
    }


if __name__ == "__main__":
    try:
        uvicorn.run("unsafe_image_classifier:app",
                    host="127.0.0.1",
                    port=3000,
                    reload=True,
                    log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
