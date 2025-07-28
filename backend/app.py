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
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional MongoDB with GridFS
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from motor.motor_asyncio import AsyncIOMotorGridFSBucket
    import gridfs
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI setup ---
app = FastAPI(title="CNN Image Classifier API with MongoDB Atlas Storage", version="2.0.0")

# CORS middleware - Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
)

# --- Model and class names ---
MODEL_PATH = "models/model.h5"
CLASS_NAMES = ["cat", "dog"]  # Adjust based on your CNN model classes
model = None

# --- MongoDB Atlas setup ---
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://sanjeevnidhir05:@123sanjeevni@cluster0.csdc7wq.mongodb.net/?retryWrites=true&w=majority")
DATABASE_NAME = os.getenv("DATABASE_NAME", "image_classify")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dogs & cats")
IMAGES_COLLECTION = os.getenv("IMAGES_COLLECTION", "uploaded_images")
DATASET_COLLECTION = os.getenv("DATASET_COLLECTION", "dataset_images")

mongodb_client = None
db = None
collection = None
images_collection = None
dataset_collection = None
fs_bucket = None
dataset_fs_bucket = None

# --- Performance metrics ---
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
    "start_time": datetime.now()
}

# --- Utility functions ---
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
        "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
        "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
        "disk_usage_percent": psutil.disk_usage('/').percent
    }

async def connect_to_mongo():
    global mongodb_client, db, collection, images_collection, dataset_collection, fs_bucket, dataset_fs_bucket
    if not MONGODB_AVAILABLE:
        logger.info("MongoDB not available - running without database")
        return False
    try:
        logger.info(f"🔄 Connecting to MongoDB Atlas...")
        logger.info(f"📍 Database: {DATABASE_NAME}")
        
        # Create client with Atlas connection string
        mongodb_client = AsyncIOMotorClient(MONGODB_URL)
        
        # Test the connection
        await mongodb_client.admin.command('ping')
        logger.info("✅ MongoDB Atlas ping successful")
        
        db = mongodb_client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        images_collection = db[IMAGES_COLLECTION]
        dataset_collection = db[DATASET_COLLECTION]
        
        # GridFS buckets
        fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="images")
        dataset_fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="dataset_images")
        
        # Create indexes
        await collection.create_index("timestamp")
        await collection.create_index("predicted_class")
        await collection.create_index("image_id")
        await images_collection.create_index("timestamp")
        await images_collection.create_index("predicted_class")
        await dataset_collection.create_index("class")
        await dataset_collection.create_index("filename")
        await dataset_collection.create_index("import_time")
        
        # Get dataset image count
        performance_metrics["dataset_images_count"] = await dataset_collection.count_documents({})
        
        logger.info("✅ MongoDB Atlas connected successfully with GridFS")
        logger.info(f"📊 Dataset images in database: {performance_metrics['dataset_images_count']}")
        logger.info(f"🏷️  Collections: {COLLECTION_NAME}, {IMAGES_COLLECTION}, {DATASET_COLLECTION}")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB Atlas connection failed: {e}")
        logger.error("Please check:")
        logger.error("  1. Your connection string and password")
        logger.error("  2. Network access whitelist in Atlas")
        logger.error("  3. Database user permissions")
        return False

async def store_image_to_gridfs(image_data, filename, metadata, bucket_type="uploads"):
    """Store image in GridFS and return the file_id"""
    target_bucket = fs_bucket if bucket_type == "uploads" else dataset_fs_bucket
    
    if target_bucket is None:
        return None
    try:
        storage_start = time.time()
        
        # Store image in GridFS
        file_id = await target_bucket.upload_from_stream(
            filename,
            io.BytesIO(image_data),
            metadata=metadata
        )
        
        storage_time = time.time() - storage_start
        performance_metrics["image_storage_times"].append(storage_time)
        if bucket_type == "uploads":
            performance_metrics["images_stored"] += 1
        
        logger.info(f"📁 Image stored in GridFS ({bucket_type}) with ID: {file_id}")
        return str(file_id)
    except Exception as e:
        logger.error(f"Failed to store image in GridFS: {e}")
        return None

async def store_prediction_to_mongodb(prediction_data, image_id=None):
    if collection is None:
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
            "image_id": image_id,  # GridFS file ID
            "system_metrics": get_system_metrics(),
            "model_info": prediction_data.get("model_info", {})
        }
        
        result = await collection.insert_one(document)
        
        # Also store in images collection for easy querying
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

# --- CNN Model functions ---
def preprocess_image_for_cnn(img_data):
    """Preprocess image for CNN model prediction"""
    try:
        # Open image and convert to RGB
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        logger.info(f"🖼️  Original image size: {img.size}")
        
        # Resize image for CNN model (adjust based on your model's input size)
        img_resized = img.resize((150, 150))  # Change to match your CNN model input
        img_array = image.img_to_array(img_resized) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        logger.info(f"📐 Processed image shape: {img_array.shape}")
        
        return img_array, img
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

def predict_with_cnn(img_array):
    """Make prediction using CNN model"""
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))
        
        logger.info(f"🧠 CNN Predictions: {predictions[0]}")
        logger.info(f"🎯 Predicted class: {predicted_class} (index: {predicted_class_idx})")
        logger.info(f"📊 Confidence: {confidence:.4f}")
        
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        logger.error(f"❌ CNN model prediction failed: {e}")
        raise HTTPException(status_code=500, detail="CNN model prediction failed")

# --- Startup and shutdown events ---
@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info(f"🔄 Loading CNN model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        logger.info("✅ CNN Model loaded successfully")
        logger.info(f"📊 Model input shape: {model.input_shape}")
        logger.info(f"🏷️  Model classes: {CLASS_NAMES}")
        
        # Test model with dummy data
        dummy_input = np.random.random((1, 150, 150, 3))
        test_pred = model.predict(dummy_input, verbose=0)
        logger.info(f"🧪 Model test successful - output shape: {test_pred.shape}")
        
    except FileNotFoundError:
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        logger.error("Please ensure your trained CNN model is saved at the correct path")
        raise HTTPException(status_code=500, detail=f"CNN Model file not found: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load CNN model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load CNN model")
    
    # Try to connect to MongoDB Atlas
    if MONGODB_AVAILABLE:
        mongodb_connected = await connect_to_mongo()
        if mongodb_connected:
            logger.info("✅ MongoDB Atlas connected - predictions and images will be stored")
        else:
            logger.warning("⚠️  MongoDB Atlas not connected - predictions and images won't be stored")
    else:
        logger.info("ℹ️  MongoDB not available - install 'motor' for database features")

@app.on_event("shutdown")
async def shutdown_event():
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB Atlas connection closed")

# --- Endpoints ---
@app.get("/")
def home():
    uptime = datetime.now() - performance_metrics["start_time"]
    return {
        "message": "Welcome to CNN Image Classifier API with MongoDB Atlas Storage",
        "status": "healthy",
        "uptime_seconds": uptime.total_seconds(),
        "model_loaded": model is not None,
        "model_info": {
            "path": MODEL_PATH,
            "input_shape": str(model.input_shape) if model is not None else "N/A",
            "classes": CLASS_NAMES,
            "total_classes": len(CLASS_NAMES)
        },
        "mongodb_connected": collection is not None if MONGODB_AVAILABLE else "disabled",
        "gridfs_enabled": fs_bucket is not None,
        "dataset_gridfs_enabled": dataset_fs_bucket is not None,
        "database_info": {
            "available": MONGODB_AVAILABLE,
            "connected": collection is not None,
            "type": "MongoDB Atlas",
            "database": DATABASE_NAME if MONGODB_AVAILABLE else "N/A"
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
    
    try:
        logger.info(f"📥 Received image file: {file.filename} ({file.content_type})")

        # Validate file type
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload PNG, JPG, or JPEG files."
            )

        # Read and validate image
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"📁 File size: {file_size / 1024:.1f} KB")
        
        # Process image for CNN model
        img_array, original_img = preprocess_image_for_cnn(contents)

        # CNN Model inference
        inference_start = time.time()
        predicted_class, confidence, raw_predictions = predict_with_cnn(img_array)
        inference_time = time.time() - inference_start

        # Store image in GridFS (MongoDB Atlas)
        image_id = None
        if MONGODB_AVAILABLE and fs_bucket is not None:
            try:
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
                logger.info(f"💾 Image stored in Atlas with ID: {image_id}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to store image in Atlas: {e}")

        total_processing_time = time.time() - request_start_time

        prediction_data = {
            "class": predicted_class,
            "confidence": confidence,
            "processing_time": total_processing_time,
            "inference_time": inference_time,
            "file_size": file_size,
            "filename": file.filename,
            "image_id": image_id,  # GridFS file ID
            "raw_predictions": raw_predictions.tolist(),  # All class probabilities
            "model_info": {
                "input_shape": str(model.input_shape),
                "classes": CLASS_NAMES,
                "model_path": MODEL_PATH
            }
        }

        # Store prediction metadata to MongoDB Atlas
        if MONGODB_AVAILABLE and collection is not None:
            try:
                asyncio.create_task(store_prediction_to_mongodb(prediction_data, image_id))
                logger.info("💾 Prediction metadata stored to MongoDB Atlas")
            except Exception as e:
                logger.warning(f"⚠️  Failed to store prediction metadata: {e}")

        # Update performance metrics
        performance_metrics["request_times"].append(total_processing_time)
        performance_metrics["model_inference_times"].append(inference_time)
        performance_metrics["successful_predictions"] += 1

        logger.info(f"✅ CNN Prediction successful: {predicted_class} ({confidence:.2%}) in {total_processing_time:.3f}s")

        return {
            "status": "success",
            "prediction": prediction_data
        }

    except HTTPException:
        raise
    except Exception as e:
        performance_metrics["failed_predictions"] += 1
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/images/")
async def get_stored_images(limit: int = 10, skip: int = 0, source: str = "all"):
    """Get list of stored images with metadata"""
    if not MONGODB_AVAILABLE or images_collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Build query filter
        query = {}
        if source != "all":
            query["source"] = source
            
        cursor = images_collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        images = []
        async for doc in cursor:
            # Convert ObjectId to string
            doc["_id"] = str(doc["_id"])
            if "prediction_id" in doc:
                doc["prediction_id"] = str(doc["prediction_id"])
            images.append(doc)
        
        total_count = await images_collection.count_documents(query)
        
        return {
            "images": images,
            "total_count": total_count,
            "limit": limit,
            "skip": skip,
            "source_filter": source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch images: {str(e)}")

@app.get("/dataset/images/")
async def get_dataset_images(class_name: str = None, limit: int = 10, skip: int = 0):
    """Get dataset images stored in MongoDB Atlas"""
    if not MONGODB_AVAILABLE or dataset_collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Build query
        query = {}
        if class_name:
            query["class"] = class_name.lower()
            
        cursor = dataset_collection.find(query).sort("import_time", -1).skip(skip).limit(limit)
        images = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            images.append(doc)
        
        total_count = await dataset_collection.count_documents(query)
        
        return {
            "dataset_images": images,
            "total_count": total_count,
            "limit": limit,
            "skip": skip,
            "class_filter": class_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dataset images: {str(e)}")

@app.get("/images/{image_id}")
async def get_image_by_id(image_id: str, source: str = "uploads"):
    """Retrieve specific image from GridFS"""
    if not MONGODB_AVAILABLE:
        raise HTTPException(status_code=503, detail="GridFS not available")
    
    target_bucket = fs_bucket if source == "uploads" else dataset_fs_bucket
    if target_bucket is None:
        raise HTTPException(status_code=503, detail="GridFS bucket not available")
    
    try:
        from bson import ObjectId
        
        # Download image from GridFS
        grid_out = await target_bucket.open_download_stream(ObjectId(image_id))
        image_data = await grid_out.read()
        
        # Get metadata
        metadata = grid_out.metadata or {}
        content_type = metadata.get("content_type", "image/jpeg")
        
        # Convert to base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "image_id": image_id,
            "image_data": image_base64,
            "content_type": content_type,
            "metadata": metadata,
            "size": len(image_data),
            "source": source
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {str(e)}")

@app.get("/metrics/")
async def get_performance_metrics():
    uptime = datetime.now() - performance_metrics["start_time"]
    request_times = list(performance_metrics["request_times"])
    inference_times = list(performance_metrics["model_inference_times"])
    db_times = list(performance_metrics["db_write_times"])
    storage_times = list(performance_metrics["image_storage_times"])

    def get_stats(times_list):
        if not times_list:
            return {"avg": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
        times_sorted = sorted(times_list)
        n = len(times_sorted)
        return {
            "avg": sum(times_sorted) / n,
            "min": min(times_sorted),
            "max": max(times_sorted),
            "p95": times_sorted[int(0.95 * n)] if n > 0 else 0,
            "p99": times_sorted[int(0.99 * n)] if n > 0 else 0
        }

    return {
        "uptime_seconds": uptime.total_seconds(),
        "total_requests": performance_metrics["total_requests"],
        "successful_predictions": performance_metrics["successful_predictions"],
        "failed_predictions": performance_metrics["failed_predictions"],
        "images_stored": performance_metrics["images_stored"],
        "dataset_images_count": performance_metrics["dataset_images_count"],
        "success_rate": (
            performance_metrics["successful_predictions"] /
            max(1, performance_metrics["total_requests"])
        ) * 100,
        "requests_per_second": performance_metrics["total_requests"] / max(1, uptime.total_seconds()),
        "response_times": get_stats(request_times),
        "inference_times": get_stats(inference_times),
        "database_write_times": get_stats(db_times),
        "image_storage_times": get_stats(storage_times),
        "system_metrics": get_system_metrics()
    }

@app.get("/predictions/stats/")
async def get_prediction_stats():
    if not MONGODB_AVAILABLE or collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Class distribution from predictions
        pipeline = [
            {
                "$group": {
                    "_id": "$predicted_class",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"},
                    "avg_processing_time": {"$avg": "$processing_time"}
                }
            }
        ]
        
        class_stats = []
        async for doc in collection.aggregate(pipeline):
            class_stats.append(doc)
        
        # Dataset class distribution
        dataset_pipeline = [
            {"$group": {"_id": "$class", "count": {"$sum": 1}}}
        ]
        
        dataset_stats = []
        if dataset_collection:
            async for doc in dataset_collection.aggregate(dataset_pipeline):
                dataset_stats.append(doc)
        
        total_predictions = await collection.count_documents({})
        total_images = await images_collection.count_documents({}) if images_collection else 0
        total_dataset_images = await dataset_collection.count_documents({}) if dataset_collection else 0
        
        return {
            "total_predictions_stored": total_predictions,
            "total_images_stored": total_images,
            "total_dataset_images": total_dataset_images,
            "prediction_class_distribution": class_stats,
            "dataset_class_distribution": dataset_stats,
            "database_status": "connected",
            "database_type": "MongoDB Atlas",
            "gridfs_status": "enabled" if fs_bucket else "disabled",
            "dataset_gridfs_status": "enabled" if dataset_fs_bucket else "disabled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model is not None else "not_loaded",
        "model_classes": CLASS_NAMES,
        "database_status": "connected" if collection is not None else "disconnected",
        "database_type": "MongoDB Atlas",
        "gridfs_status": "enabled" if fs_bucket is not None else "disabled",
        "dataset_gridfs_status": "enabled" if dataset_fs_bucket is not None else "disabled",
        "images_stored": performance_metrics["images_stored"],
        "dataset_images_count": performance_metrics["dataset_images_count"]
    }

# Test endpoint for CNN model
@app.post("/test-model/")
async def test_model():
    """Test CNN model with dummy data"""
    if model is None:
        raise HTTPException(status_code=503, detail="CNN Model not loaded")
    
    try:
        # Create dummy image data
        dummy_image = np.random.random((1, 150, 150, 3))
        
        # Make prediction
        start_time = time.time()
        predictions = model.predict(dummy_image, verbose=0)
        inference_time = time.time() - start_time
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))
        
        return {
            "status": "success",
            "message": "CNN Model test successful",
            "test_results": {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "inference_time": inference_time,
                "raw_predictions": predictions[0].tolist(),
                "model_shape": str(model.input_shape),
                "classes": CLASS_NAMES
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model test failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)