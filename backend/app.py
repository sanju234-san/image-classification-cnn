from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime
import logging
import asyncio
from collections import deque
import os

# Optional system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional MongoDB with GridFS
try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- FastAPI setup
app = FastAPI(title="CNN Image Classifier API with MongoDB Compass Storage", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Model and classes
MODEL_PATH = "models/model.h5"
CLASS_NAMES = ["cat", "dog"] # Adapt this as needed for your trained model
model = None

# ---- MongoDB Compass/local config
MONGODB_CONNECTION_STRING = "mongodb://localhost:27017/"
MONGODB_DATABASE = "image_classifier"

COLLECTION_NAME = "predictions"
IMAGES_COLLECTION = "uploaded_images"
DATASET_COLLECTION = "cats"
DOGS_COLLECTION = "dogs"  # If you also leverage this collection

# Global MongoDB holders
mongodb_client = None
db = None
collection = None
images_collection = None
dataset_collection = None
fs_bucket = None
dataset_fs_bucket = None

mongodb_connection_status = {
    "connected": False,
    "error": None,
    "last_attempt": None,
    "retry_count": 0,
}

performance_metrics = {
    "request_times": deque(maxlen=1000),
    "model_inference_times": deque(maxlen=1000),
    "db_write_times": deque(maxlen=1000),
    "image_storage_times": deque(maxlen=1000),
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "images_stored": 0,
    "dataset_images_count": 0,
    "start_time": datetime.now(),
}

def get_system_metrics():
    if not PSUTIL_AVAILABLE:
        return {
            "cpu_percent": "N/A",
            "memory_percent": "N/A",
            "memory_used_mb": "N/A",
            "memory_available_mb": "N/A",
            "disk_usage_percent": "N/A"
        }
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_mb": round(psutil.virtual_memory().used / 1024 / 1024, 2),
        "memory_available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
        "disk_usage_percent": psutil.disk_usage('/').percent
    }

async def connect_to_mongo():
    global mongodb_client, db, collection, images_collection, dataset_collection, fs_bucket, dataset_fs_bucket
    mongodb_connection_status["last_attempt"] = datetime.now()
    mongodb_connection_status["retry_count"] += 1
    if not MONGODB_AVAILABLE:
        mongodb_connection_status["error"] = "Motor driver not available"
        logger.error("MongoDB motor driver not available")
        return False
    try:
        logger.info("Connecting to MongoDB Compass/local ...")
        mongodb_client = AsyncIOMotorClient(MONGODB_CONNECTION_STRING)
        db = mongodb_client[MONGODB_DATABASE]
        collection = db[COLLECTION_NAME]
        images_collection = db[IMAGES_COLLECTION]
        dataset_collection = db[DATASET_COLLECTION]
        fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="images")
        dataset_fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="dataset_images")
        try:
            await collection.create_index("timestamp")
            await collection.create_index("predicted_class")
            await collection.create_index("image_id")
            await images_collection.create_index("timestamp")
            await images_collection.create_index("predicted_class")
            await dataset_collection.create_index("filename")
            await dataset_collection.create_index("uploadDate")
            try:
                dogs_collection = db[DOGS_COLLECTION]
                await dogs_collection.create_index("filename")
                await dogs_collection.create_index("uploadDate")
            except Exception:
                pass
        except Exception as idx_error:
            logger.warning(f"Index creation warning: {idx_error}")
        try:
            performance_metrics["dataset_images_count"] = await dataset_collection.count_documents({})
        except Exception:
            performance_metrics["dataset_images_count"] = 0
        mongodb_connection_status["connected"] = True
        mongodb_connection_status["error"] = None
        logger.info("MongoDB local connected successfully")
        return True
    except Exception as e:
        mongodb_connection_status["error"] = f"MongoDB connection failed: {str(e)}"
        mongodb_connection_status["connected"] = False
        logger.error(f"MongoDB connection failed: {e}")
        return False

async def store_image_to_gridfs(image_data, filename, metadata, bucket_type="uploads"):
    if not mongodb_connection_status["connected"]:
        return None
    target_bucket = fs_bucket if bucket_type == "uploads" else dataset_fs_bucket
    if target_bucket is None:
        return None
    try:
        storage_start = time.time()
        file_id = await target_bucket.upload_from_stream(
            filename,
            io.BytesIO(image_data),
            metadata=metadata
        )
        storage_time = time.time() - storage_start
        performance_metrics["image_storage_times"].append(storage_time)
        if bucket_type == "uploads":
            performance_metrics["images_stored"] += 1
        return str(file_id)
    except Exception as e:
        logger.error(f"Failed to store image in GridFS: {e}")
        return None

async def store_prediction_to_mongodb(prediction_data, image_id=None):
    if not mongodb_connection_status["connected"] or collection is None:
        return False
    try:
        db_start = time.time()
        document = {
            "timestamp": datetime.now(),
            "predicted_class": prediction_data["class"],
            "confidence": prediction_data["confidence"],
            "processing_time": prediction_data.get("processing_time", 0),
            "inference_time": prediction_data.get("inference_time", 0),
            "file_size": prediction_data.get("file_size", 0),
            "filename": prediction_data.get("filename", "unknown"),
            "image_id": image_id,
            "raw_predictions": prediction_data.get("raw_predictions", []),
            "system_metrics": get_system_metrics(),
            "model_info": prediction_data.get("model_info", {})
        }
        result = await collection.insert_one(document)
        if images_collection is not None:
            image_doc = {
                "timestamp": datetime.now(),
                "filename": prediction_data.get("filename", "unknown"),
                "predicted_class": prediction_data["class"],
                "confidence": prediction_data["confidence"],
                "file_size": prediction_data.get("file_size", 0),
                "image_gridfs_id": image_id,
                "prediction_id": str(result.inserted_id),
                "source": "user_upload"
            }
            await images_collection.insert_one(image_doc)
        db_time = time.time() - db_start
        performance_metrics["db_write_times"].append(db_time)
        return True
    except Exception as e:
        logger.error(f"Failed to store prediction: {e}")
        return False

# --- CNN Model functions
def preprocess_image_for_cnn(img_data):
    try:
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_resized = img.resize((150, 150))  # Change if your model input shape is different
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

def predict_with_cnn(img_array):
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        logger.error(f"CNN model prediction failed: {e}")
        raise HTTPException(status_code=500, detail="CNN model prediction failed")

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        logger.error(f"Cannot load CNN model: {e}")
        model = None
    if MONGODB_AVAILABLE:
        await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()

@app.get("/")
def home():
    uptime = datetime.now() - performance_metrics["start_time"]
    return {
        "message": "Welcome to CNN Image Classifier API with MongoDB Compass/local Storage",
        "status": "healthy",
        "uptime_seconds": round(uptime.total_seconds(), 2),
        "model_loaded": model is not None,
        "model_info": {
            "path": MODEL_PATH,
            "input_shape": str(model.input_shape) if model is not None else "N/A",
            "classes": CLASS_NAMES,
            "total_classes": len(CLASS_NAMES)
        },
        "database_info": {
            "available": MONGODB_AVAILABLE,
            "connected": mongodb_connection_status["connected"],
            "type": "local",
            "database": MONGODB_DATABASE,
            "collections": {
                "main": COLLECTION_NAME,
                "images": IMAGES_COLLECTION,
                "dataset_cats": DATASET_COLLECTION,
                "dataset_dogs": DOGS_COLLECTION
            },
            "last_connection_attempt": mongodb_connection_status["last_attempt"].isoformat() if mongodb_connection_status["last_attempt"] else None,
            "retry_count": mongodb_connection_status["retry_count"],
            "error": mongodb_connection_status["error"]
        },
        "gridfs_status": {
            "uploads": fs_bucket is not None,
            "dataset": dataset_fs_bucket is not None
        },
        "performance": {
            "total_requests": performance_metrics["total_requests"],
            "successful_predictions": performance_metrics["successful_predictions"],
            "failed_predictions": performance_metrics["failed_predictions"],
            "images_stored": performance_metrics["images_stored"],
            "dataset_images_count": performance_metrics["dataset_images_count"]
        }
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    request_start_time = time.time()
    performance_metrics["total_requests"] += 1
    if model is None:
        raise HTTPException(status_code=503, detail="CNN Model not loaded")
    try:
        contents = await file.read()
        file_size = len(contents)
        img_array, original_img = preprocess_image_for_cnn(contents)
        inference_start = time.time()
        predicted_class, confidence, raw_predictions = predict_with_cnn(img_array)
        inference_time = time.time() - inference_start
        image_id = None
        if mongodb_connection_status["connected"] and fs_bucket is not None:
            image_metadata = {
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": file_size,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "upload_time": datetime.now(),
                "original_size": f"{original_img.size[0]}x{original_img.size[1]}",
                "source": "user_upload"
            }
            image_id = await store_image_to_gridfs(contents, file.filename, image_metadata, "uploads")
        total_processing_time = time.time() - request_start_time
        prediction_data = {
            "class": predicted_class,
            "confidence": confidence,
            "processing_time": total_processing_time,
            "inference_time": inference_time,
            "file_size": file_size,
            "filename": file.filename,
            "image_id": image_id,
            "raw_predictions": raw_predictions.tolist(),
            "model_info": {
                "input_shape": str(model.input_shape),
                "classes": CLASS_NAMES,
                "model_path": MODEL_PATH
            }
        }
        if mongodb_connection_status["connected"]:
            asyncio.create_task(store_prediction_to_mongodb(prediction_data, image_id))
        performance_metrics["request_times"].append(total_processing_time)
        performance_metrics["model_inference_times"].append(inference_time)
        performance_metrics["successful_predictions"] += 1
        return {
            "status": "success",
            "prediction": prediction_data,
            "database_stored": mongodb_connection_status["connected"]
        }
    except HTTPException:
        raise
    except Exception as e:
        performance_metrics["failed_predictions"] += 1
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ...remaning endpoints from your original unchanged, except that all Atlas/cloud-related code is gone...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
