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
import urllib.parse

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
    from bson import ObjectId
    MONGODB_AVAILABLE = True
    print("✅ MongoDB motor driver available")
except ImportError:
    MONGODB_AVAILABLE = False
    print("❌ MongoDB motor driver not available - install with: pip install motor pymongo")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI setup ---
app = FastAPI(title="CNN Image Classifier API with MongoDB Atlas Storage", version="2.1.0")

# CORS middleware - Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.onrender.com",
        "*"  # Allow all origins for deployment
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
)

# --- Model and class names ---
MODEL_PATH = "models/model.h5"
CLASS_NAMES = ["cat", "dog"]  # Adjust based on your CNN model classes
model = None

# --- MongoDB Atlas Configuration ---
# Updated MongoDB Atlas connection string
MONGODB_CONNECTION_STRING = "mongodb+srv://sanjeevnidhir05:@123sanjeevni@cluster0.csdc7wq.mongodb.net/"

# Based on your MongoDB Atlas interface, using the correct database and collection names
MONGODB_USERNAME = "sanjeevnidhir05"
MONGODB_PASSWORD = "@123sanjeevni"
MONGODB_CLUSTER = "cluster0.csdc7wq.mongodb.net"
MONGODB_DATABASE = "image_classify"  # Updated to match your Atlas database name

# Properly URL encode credentials to handle special characters
def safe_url_encode(value):
    """Safely encode URL components, handling None values"""
    if value is None:
        return ""
    return urllib.parse.quote_plus(str(value))

encoded_username = safe_url_encode(MONGODB_USERNAME)
encoded_password = safe_url_encode(MONGODB_PASSWORD)

print(f"🔐 Original username: {MONGODB_USERNAME}")
print(f"🔐 Encoded username: {encoded_username}")
print(f"🔐 Password length: {len(MONGODB_PASSWORD)} characters")
print(f"🔐 Encoded password length: {len(encoded_password)} characters")

# Construct MongoDB Atlas connection string with proper encoding
MONGODB_URL = f"mongodb+srv://{encoded_username}:{encoded_password}@{MONGODB_CLUSTER}/{MONGODB_DATABASE}?retryWrites=true&w=majority&authSource=admin"

print(f"🔗 MongoDB Atlas URL: mongodb+srv://{encoded_username}:****@{MONGODB_CLUSTER}/{MONGODB_DATABASE}")

# Database and collection names - Updated based on your MongoDB Atlas interface
DATABASE_NAME = MONGODB_DATABASE  # "image_classify"
COLLECTION_NAME = "predictions"  # For storing prediction metadata
IMAGES_COLLECTION = "uploaded_images"  # For storing uploaded image metadata
DATASET_COLLECTION = "cats"  # Based on your Atlas interface showing "cats" collection
# Additional collections that might exist based on your setup
DOGS_COLLECTION = "dogs"  # In case you have a separate dogs collection

# Global MongoDB variables
mongodb_client = None
db = None
collection = None
images_collection = None
dataset_collection = None
fs_bucket = None
dataset_fs_bucket = None

# Connection status tracking
mongodb_connection_status = {
    "connected": False,
    "error": None,
    "last_attempt": None,
    "retry_count": 0
}

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
        "memory_used_mb": round(psutil.virtual_memory().used / 1024 / 1024, 2),
        "memory_available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
        "disk_usage_percent": psutil.disk_usage('/').percent
    }

async def test_mongodb_atlas_connection():
    """Test MongoDB Atlas connection with detailed error reporting"""
    if not MONGODB_AVAILABLE:
        return False, "MongoDB motor driver not available"
    
    try:
        logger.info("🔄 Testing MongoDB Atlas connection...")
        logger.info(f"📍 Target Database: {DATABASE_NAME}")
        logger.info(f"🔗 Cluster: {MONGODB_CLUSTER}")
        
        # Create test client with proper Atlas configuration
        test_client = AsyncIOMotorClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=15000,  # 15 second timeout
            connectTimeoutMS=15000,
            socketTimeoutMS=15000,
            maxPoolSize=10,
            retryWrites=True,
            ssl=True,  # Required for Atlas
            tlsAllowInvalidCertificates=False,
            authSource='admin'  # Specify auth source
        )
        
        # Test the connection with admin ping
        logger.info("🏓 Testing connection with ping...")
        await test_client.admin.command('ping')
        logger.info("✅ Ping successful")
        
        # Test database access
        test_db = test_client[DATABASE_NAME]
        collections = await test_db.list_collection_names()
        logger.info(f"📋 Available collections: {collections}")
        
        # Test write permission
        test_collection = test_db[COLLECTION_NAME]
        test_doc = {
            "test": True,
            "timestamp": datetime.now(),
            "connection_test": "successful"
        }
        test_result = await test_collection.insert_one(test_doc)
        logger.info(f"✏️ Test write successful, ID: {test_result.inserted_id}")
        
        # Clean up test document
        await test_collection.delete_one({"_id": test_result.inserted_id})
        logger.info("🧹 Test document cleaned up")
        
        test_client.close()
        logger.info("✅ MongoDB Atlas connection test successful")
        return True, f"Connection successful. Collections: {collections}"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ MongoDB Atlas connection test failed: {error_msg}")
        
        # Provide specific error guidance
        if "authentication failed" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return False, f"Authentication failed - Check username/password: {error_msg}"
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return False, f"Connection timeout - Check network access in Atlas: {error_msg}"
        elif "name or service not known" in error_msg.lower():
            return False, f"DNS resolution failed - Check cluster URL: {error_msg}"
        elif "not authorized" in error_msg.lower():
            return False, f"Authorization failed - Check database permissions: {error_msg}"
        elif "rfc 3986" in error_msg.lower() or "must be escaped" in error_msg.lower():
            return False, f"URL encoding error - Credentials need proper encoding: {error_msg}"
        else:
            return False, f"Connection error: {error_msg}"

async def connect_to_mongo():
    """Connect to MongoDB Atlas with retry logic"""
    global mongodb_client, db, collection, images_collection, dataset_collection, fs_bucket, dataset_fs_bucket
    
    mongodb_connection_status["last_attempt"] = datetime.now()
    mongodb_connection_status["retry_count"] += 1
    
    if not MONGODB_AVAILABLE:
        error_msg = "MongoDB motor driver not available - install with: pip install motor pymongo"
        mongodb_connection_status["error"] = error_msg
        logger.error(f"❌ {error_msg}")
        return False
    
    try:
        logger.info("🔄 Connecting to MongoDB Atlas...")
        logger.info(f"📍 Database: {DATABASE_NAME}")
        logger.info(f"🔗 Cluster: {MONGODB_CLUSTER}")
        logger.info(f"👤 Username: {MONGODB_USERNAME}")
        logger.info(f"🔄 Retry attempt: {mongodb_connection_status['retry_count']}")
        
        # Test connection first
        connection_ok, test_error = await test_mongodb_atlas_connection()
        if not connection_ok:
            mongodb_connection_status["error"] = test_error
            logger.error(f"❌ Connection test failed: {test_error}")
            return False
        
        # Create the actual client with Atlas configuration
        mongodb_client = AsyncIOMotorClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=30000,  # 30 second timeout
            connectTimeoutMS=20000,
            socketTimeoutMS=20000,
            maxPoolSize=50,
            minPoolSize=5,
            retryWrites=True,
            w="majority",
            ssl=True,  # Required for Atlas
            tlsAllowInvalidCertificates=False,
            authSource='admin'
        )
        
        # Verify connection
        await mongodb_client.admin.command('ping')
        logger.info("✅ MongoDB Atlas ping successful")
        
        # Set up database and collections
        db = mongodb_client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        images_collection = db[IMAGES_COLLECTION]
        dataset_collection = db[DATASET_COLLECTION]
        
        logger.info(f"📁 Using collections:")
        logger.info(f"   - Main: {COLLECTION_NAME}")
        logger.info(f"   - Images: {IMAGES_COLLECTION}")
        logger.info(f"   - Dataset: {DATASET_COLLECTION}")
        
        # Set up GridFS buckets
        fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="images")
        dataset_fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="dataset_images")
        
        # Create indexes for better performance
        try:
            await collection.create_index("timestamp")
            await collection.create_index("predicted_class")
            await collection.create_index("image_id")
            await images_collection.create_index("timestamp")
            await images_collection.create_index("predicted_class")
            await dataset_collection.create_index("filename")
            await dataset_collection.create_index("uploadDate")  # GridFS standard field
            # If you have a dogs collection as well
            try:
                dogs_collection = db["dogs"]
                await dogs_collection.create_index("filename")
                await dogs_collection.create_index("uploadDate")
            except Exception:
                pass  # Dogs collection might not exist
            logger.info("✅ Database indexes created")
        except Exception as idx_error:
            logger.warning(f"⚠️  Index creation warning: {idx_error}")
        
        # Get dataset image count
        try:
            performance_metrics["dataset_images_count"] = await dataset_collection.count_documents({})
        except Exception as count_error:
            logger.warning(f"⚠️  Could not count dataset images: {count_error}")
            performance_metrics["dataset_images_count"] = 0
        
        mongodb_connection_status["connected"] = True
        mongodb_connection_status["error"] = None
        
        logger.info("✅ MongoDB Atlas connected successfully!")
        logger.info(f"📊 Dataset images in database: {performance_metrics['dataset_images_count']}")
        logger.info("🗄️  GridFS buckets initialized for image storage")
        
        return True
        
    except Exception as e:
        error_msg = f"MongoDB Atlas connection failed: {str(e)}"
        mongodb_connection_status["error"] = error_msg
        mongodb_connection_status["connected"] = False
        
        logger.error(f"❌ {error_msg}")
        logger.error("🔧 Troubleshooting steps:")
        logger.error("  1. Check MongoDB Atlas cluster is running")
        logger.error("  2. Verify network access allows all IPs (0.0.0.0/0)")
        logger.error("  3. Check database user has read/write permissions")
        logger.error("  4. Verify username and password are correct")
        logger.error("  5. Ensure cluster URL is correct")
        
        return False

async def store_image_to_gridfs(image_data, filename, metadata, bucket_type="uploads"):
    """Store image in GridFS and return the file_id"""
    if not mongodb_connection_status["connected"]:
        logger.warning("⚠️  MongoDB not connected - skipping image storage")
        return None
        
    target_bucket = fs_bucket if bucket_type == "uploads" else dataset_fs_bucket
    
    if target_bucket is None:
        logger.warning("⚠️  GridFS bucket not available")
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
        
        logger.info(f"📁 Image stored in GridFS ({bucket_type}) with ID: {file_id} in {storage_time:.3f}s")
        return str(file_id)
        
    except Exception as e:
        logger.error(f"❌ Failed to store image in GridFS: {e}")
        return None

async def store_prediction_to_mongodb(prediction_data, image_id=None):
    """Store prediction metadata to MongoDB"""
    if not mongodb_connection_status["connected"] or collection is None:
        logger.warning("⚠️  MongoDB not connected - skipping prediction storage")
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
            "raw_predictions": prediction_data.get("raw_predictions", []),
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
        
        logger.info(f"💾 Prediction stored to MongoDB in {db_time:.3f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to store prediction: {e}")
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
    
    # Load CNN model
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
        # Don't raise error - allow API to start without model for testing
        model = None
    except Exception as e:
        logger.error(f"❌ Failed to load CNN model: {e}")
        model = None
    
    # Connect to MongoDB Atlas
    if MONGODB_AVAILABLE:
        logger.info("🔄 Attempting MongoDB Atlas connection...")
        connection_success = await connect_to_mongo()
        
        if connection_success:
            logger.info("✅ MongoDB Atlas connected - full functionality available")
        else:
            logger.warning("⚠️  MongoDB Atlas connection failed - API will work without database storage")
            logger.info("🔧 You can retry connection using the /reconnect-db/ endpoint")
    else:
        logger.warning("⚠️  MongoDB dependencies not available")

@app.on_event("shutdown")
async def shutdown_event():
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        logger.info("🔌 MongoDB Atlas connection closed")

# --- Endpoints ---
@app.get("/")
def home():
    uptime = datetime.now() - performance_metrics["start_time"]
    return {
        "message": "Welcome to CNN Image Classifier API with MongoDB Atlas Storage",
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
            "type": "MongoDB Atlas",
            "cluster": MONGODB_CLUSTER,
            "database": DATABASE_NAME,
            "collections": {
                "main": COLLECTION_NAME,  # predictions
                "images": IMAGES_COLLECTION,  # uploaded_images
                "dataset_cats": DATASET_COLLECTION,  # cats
                "dataset_dogs": DOGS_COLLECTION  # dogs (if exists)
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

@app.post("/reconnect-db/")
async def reconnect_database():
    """Manually retry MongoDB Atlas connection"""
    logger.info("🔄 Manual database reconnection requested...")
    
    # Reset connection status
    mongodb_connection_status["retry_count"] = 0
    
    success = await connect_to_mongo()
    
    return {
        "status": "success" if success else "failed",
        "connected": mongodb_connection_status["connected"],
        "error": mongodb_connection_status["error"],
        "retry_count": mongodb_connection_status["retry_count"],
        "timestamp": datetime.now().isoformat(),
        "database": DATABASE_NAME,
        "cluster": MONGODB_CLUSTER
    }

@app.get("/test-connection/")
async def test_mongodb_connection_endpoint():
    """Test MongoDB Atlas connection with detailed diagnostics"""
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "mongodb_available": MONGODB_AVAILABLE,
        "connection_info": {
            "username": MONGODB_USERNAME,
            "cluster": MONGODB_CLUSTER,
            "database": MONGODB_DATABASE,
            "connection_string_sample": f"mongodb+srv://{encoded_username}:****@{MONGODB_CLUSTER}/{MONGODB_DATABASE}",
        },
        "encoding_info": {
            "original_username_length": len(MONGODB_USERNAME),
            "encoded_username_length": len(encoded_username),
            "original_password_length": len(MONGODB_PASSWORD),
            "encoded_password_length": len(encoded_password),
            "password_has_special_chars": any(c in MONGODB_PASSWORD for c in "@!#$%^&*()+=[]{}|;:,.<>?")
        }
    }
    
    if not MONGODB_AVAILABLE:
        return {
            "status": "error",
            "message": "MongoDB motor driver not available",
            "diagnostics": diagnostics,
            "solution": "Run: pip install motor pymongo"
        }
    
    try:
        # Test connection
        connection_ok, error_msg = await test_mongodb_atlas_connection()
        
        diagnostics.update({
            "connection_test": "passed" if connection_ok else "failed",
            "error_message": error_msg,
            "current_status": mongodb_connection_status
        })
        
        if connection_ok:
            return {
                "status": "success",
                "message": "MongoDB Atlas connection successful!",
                "diagnostics": diagnostics
            }
        else:
            return {
                "status": "error", 
                "message": error_msg,
                "diagnostics": diagnostics,
                "troubleshooting": [
                    "1. Check MongoDB Atlas cluster is running",
                    "2. Verify network access allows all IPs (0.0.0.0/0) in Atlas dashboard",
                    "3. Confirm database user exists with read/write permissions",
                    "4. Check username/password are correct",
                    "5. Ensure cluster URL is correct",
                    "6. Try connecting with MongoDB Compass using the same credentials"
                ]
            }
            
    except Exception as e:
        diagnostics["connection_error"] = str(e)
        return {
            "status": "error",
            "message": f"Connection test failed: {str(e)}",
            "diagnostics": diagnostics
        }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    request_start_time = time.time()
    performance_metrics["total_requests"] += 1
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="CNN Model not loaded")
    
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
        if mongodb_connection_status["connected"] and fs_bucket is not None:
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
                if image_id:
                    logger.info(f"💾 Image stored in Atlas GridFS with ID: {image_id}")
                
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
        if mongodb_connection_status["connected"]:
            try:
                asyncio.create_task(store_prediction_to_mongodb(prediction_data, image_id))
            except Exception as e:
                logger.warning(f"⚠️  Failed to store prediction metadata: {e}")

        # Update performance metrics
        performance_metrics["request_times"].append(total_processing_time)
        performance_metrics["model_inference_times"].append(inference_time)
        performance_metrics["successful_predictions"] += 1

        logger.info(f"✅ CNN Prediction successful: {predicted_class} ({confidence:.2%}) in {total_processing_time:.3f}s")

        return {
            "status": "success",
            "prediction": prediction_data,
            "database_stored": mongodb_connection_status["connected"]
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
    if not mongodb_connection_status["connected"] or images_collection is None:
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
async def get_dataset_images(limit: int = 10, skip: int = 0, class_filter: str = "all"):
    """Get list of dataset images with metadata"""
    if not mongodb_connection_status["connected"] or dataset_collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Build query filter
        query = {}
        if class_filter != "all":
            query["class"] = class_filter
            
        cursor = dataset_collection.find(query).sort("import_time", -1).skip(skip).limit(limit)
        images = []
        async for doc in cursor:
            # Convert ObjectId to string
            doc["_id"] = str(doc["_id"])
            if "gridfs_id" in doc:
                doc["gridfs_id"] = str(doc["gridfs_id"])
            images.append(doc)
        
        total_count = await dataset_collection.count_documents(query)
        
        return {
            "images": images,
            "total_count": total_count,
            "limit": limit,
            "skip": skip,
            "class_filter": class_filter
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dataset images: {str(e)}")

@app.get("/image/{image_id}")
async def get_image(image_id: str, bucket_type: str = "uploads"):
    """Retrieve image from GridFS by ID"""
    if not mongodb_connection_status["connected"]:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    target_bucket = fs_bucket if bucket_type == "uploads" else dataset_fs_bucket
    
    if target_bucket is None:
        raise HTTPException(status_code=503, detail="GridFS bucket not available")
    
    try:
        # Convert string ID to ObjectId
        try:
            file_id = ObjectId(image_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image ID format")
        
        # Get file info
        file_info = await target_bucket.find({"_id": file_id}).to_list(length=1)
        if not file_info:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Download file
        grid_out = await target_bucket.open_download_stream(file_id)
        image_data = await grid_out.read()
        
        # Get file metadata
        file_doc = file_info[0]
        content_type = file_doc.metadata.get("content_type", "image/jpeg") if file_doc.metadata else "image/jpeg"
        
        from fastapi.responses import Response
        return Response(content=image_data, media_type=content_type)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to retrieve image {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve image: {str(e)}")

@app.delete("/image/{image_id}")
async def delete_image(image_id: str, bucket_type: str = "uploads"):
    """Delete image from GridFS and related metadata"""
    if not mongodb_connection_status["connected"]:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    target_bucket = fs_bucket if bucket_type == "uploads" else dataset_fs_bucket
    
    if target_bucket is None:
        raise HTTPException(status_code=503, detail="GridFS bucket not available")
    
    try:
        # Convert string ID to ObjectId
        try:
            file_id = ObjectId(image_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image ID format")
        
        # Check if file exists
        file_info = await target_bucket.find({"_id": file_id}).to_list(length=1)
        if not file_info:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Delete from GridFS
        await target_bucket.delete(file_id)
        
        # Delete related metadata
        if bucket_type == "uploads":
            # Delete from images collection
            await images_collection.delete_many({"image_gridfs_id": image_id})
            # Delete predictions
            await collection.delete_many({"image_id": image_id})
        else:
            # Delete from dataset collection
            await dataset_collection.delete_many({"gridfs_id": image_id})
        
        logger.info(f"🗑️  Deleted image {image_id} and related metadata")
        
        return {
            "status": "success",
            "message": f"Image {image_id} deleted successfully",
            "bucket_type": bucket_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete image {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")

@app.post("/dataset/upload/")
async def upload_dataset_image(file: UploadFile = File(...), class_name: str = "unknown"):
    """Upload image to dataset collection"""
    if not mongodb_connection_status["connected"]:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        logger.info(f"📥 Uploading dataset image: {file.filename} for class: {class_name}")

        # Validate file type
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload PNG, JPG, or JPEG files."
            )

        # Read image data
        contents = await file.read()
        file_size = len(contents)
        
        # Validate class name
        if class_name not in CLASS_NAMES and class_name != "unknown":
            logger.warning(f"⚠️  Class '{class_name}' not in predefined classes: {CLASS_NAMES}")
        
        # Store image in GridFS
        image_metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": file_size,
            "class": class_name,
            "upload_time": datetime.now(),
            "source": "dataset_upload"
        }
        
        image_id = await store_image_to_gridfs(contents, file.filename, image_metadata, "dataset")
        
        if image_id:
            # Store metadata in dataset collection
            dataset_doc = {
                "filename": file.filename,
                "class": class_name,
                "file_size": file_size,
                "gridfs_id": image_id,
                "import_time": datetime.now(),
                "content_type": file.content_type,
                "source": "manual_upload"
            }
            
            result = await dataset_collection.insert_one(dataset_doc)
            performance_metrics["dataset_images_count"] += 1
            
            logger.info(f"✅ Dataset image uploaded successfully: {image_id}")
            
            return {
                "status": "success",
                "image_id": image_id,
                "metadata_id": str(result.inserted_id),
                "class": class_name,
                "filename": file.filename,
                "file_size": file_size
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store image")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Dataset upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {str(e)}")

@app.get("/predictions/")
async def get_predictions(limit: int = 10, skip: int = 0, class_filter: str = "all"):
    """Get prediction history with filters"""
    if not mongodb_connection_status["connected"] or collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Build query filter
        query = {}
        if class_filter != "all":
            query["predicted_class"] = class_filter
            
        cursor = collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        predictions = []
        async for doc in cursor:
            # Convert ObjectId to string
            doc["_id"] = str(doc["_id"])
            predictions.append(doc)
        
        total_count = await collection.count_documents(query)
        
        return {
            "predictions": predictions,
            "total_count": total_count,
            "limit": limit,
            "skip": skip,
            "class_filter": class_filter
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")

@app.get("/stats/")
async def get_statistics():
    """Get comprehensive statistics"""
    if not mongodb_connection_status["connected"]:
        stats = {
            "database_connected": False,
            "message": "Database not connected - showing limited stats"
        }
    else:
        try:
            # Database statistics
            total_predictions = await collection.count_documents({})
            total_images = await images_collection.count_documents({})
            total_dataset_images = await dataset_collection.count_documents({})
            
            # Class distribution for predictions
            prediction_pipeline = [
                {"$group": {"_id": "$predicted_class", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            prediction_class_dist = []
            async for doc in collection.aggregate(prediction_pipeline):
                prediction_class_dist.append({"class": doc["_id"], "count": doc["count"]})
            
            # Class distribution for dataset
            dataset_pipeline = [
                {"$group": {"_id": "$class", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            dataset_class_dist = []
            async for doc in dataset_collection.aggregate(dataset_pipeline):
                dataset_class_dist.append({"class": doc["_id"], "count": doc["count"]})
            
            # Recent activity
            recent_predictions = await collection.find({}).sort("timestamp", -1).limit(5).to_list(length=5)
            for pred in recent_predictions:
                pred["_id"] = str(pred["_id"])
            
            stats = {
                "database_connected": True,
                "totals": {
                    "predictions": total_predictions,
                    "uploaded_images": total_images,
                    "dataset_images": total_dataset_images
                },
                "class_distributions": {
                    "predictions": prediction_class_dist,
                    "dataset": dataset_class_dist
                },
                "recent_predictions": recent_predictions
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get database statistics: {e}")
            stats = {
                "database_connected": True,
                "error": str(e),
                "message": "Failed to fetch database statistics"
            }
    
    # Performance metrics (always available)
    uptime = datetime.now() - performance_metrics["start_time"]
    
    # Calculate averages
    avg_request_time = np.mean(performance_metrics["request_times"]) if performance_metrics["request_times"] else 0
    avg_inference_time = np.mean(performance_metrics["model_inference_times"]) if performance_metrics["model_inference_times"] else 0
    avg_db_time = np.mean(performance_metrics["db_write_times"]) if performance_metrics["db_write_times"] else 0
    avg_storage_time = np.mean(performance_metrics["image_storage_times"]) if performance_metrics["image_storage_times"] else 0
    
    stats.update({
        "performance": {
            "uptime_seconds": round(uptime.total_seconds(), 2),
            "total_requests": performance_metrics["total_requests"],
            "successful_predictions": performance_metrics["successful_predictions"],
            "failed_predictions": performance_metrics["failed_predictions"],
            "images_stored": performance_metrics["images_stored"],
            "success_rate": round(
                (performance_metrics["successful_predictions"] / max(performance_metrics["total_requests"], 1)) * 100, 2
            ),
            "average_times": {
                "request_processing": round(avg_request_time, 3),
                "model_inference": round(avg_inference_time, 3),
                "database_write": round(avg_db_time, 3),
                "image_storage": round(avg_storage_time, 3)
            }
        },
        "system": get_system_metrics(),
        "model_info": {
            "loaded": model is not None,
            "classes": CLASS_NAMES,
            "input_shape": str(model.input_shape) if model is not None else "N/A"
        }
    })
    
    return stats

@app.get("/health/")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {
            "api": {"status": "ok", "message": "API is running"},
            "model": {
                "status": "ok" if model is not None else "error",
                "message": "Model loaded successfully" if model is not None else "Model not loaded"
            },
            "database": {
                "status": "ok" if mongodb_connection_status["connected"] else "warning",
                "message": "Connected to MongoDB Atlas" if mongodb_connection_status["connected"] else "Database not connected",
                "error": mongodb_connection_status.get("error")
            },
            "gridfs": {
                "status": "ok" if (fs_bucket is not None and dataset_fs_bucket is not None) else "warning",
                "message": "GridFS buckets available" if (fs_bucket is not None and dataset_fs_bucket is not None) else "GridFS not fully initialized"
            }
        },
        "performance": {
            "total_requests": performance_metrics["total_requests"],
            "successful_predictions": performance_metrics["successful_predictions"],
            "failed_predictions": performance_metrics["failed_predictions"]
        }
    }
    
    # Determine overall status
    if model is None:
        health_status["status"] = "degraded"
    elif not mongodb_connection_status["connected"]:
        health_status["status"] = "warning"
    
    return health_status

# Additional utility endpoints
@app.get("/classes/")
async def get_classes():
    """Get available model classes"""
    return {
        "classes": CLASS_NAMES,
        "total_classes": len(CLASS_NAMES),
        "model_loaded": model is not None
    }

@app.post("/bulk-delete/")
async def bulk_delete_images(image_ids: list, bucket_type: str = "uploads"):
    """Delete multiple images at once"""
    if not mongodb_connection_status["connected"]:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    target_bucket = fs_bucket if bucket_type == "uploads" else dataset_fs_bucket
    
    if target_bucket is None:
        raise HTTPException(status_code=503, detail="GridFS bucket not available")
    
    results = {
        "deleted": [],
        "failed": [],
        "total_requested": len(image_ids)
    }
    
    for image_id in image_ids:
        try:
            # Convert string ID to ObjectId
            file_id = ObjectId(image_id)
            
            # Check if file exists
            file_info = await target_bucket.find({"_id": file_id}).to_list(length=1)
            if file_info:
                # Delete from GridFS
                await target_bucket.delete(file_id)
                
                # Delete related metadata
                if bucket_type == "uploads":
                    await images_collection.delete_many({"image_gridfs_id": image_id})
                    await collection.delete_many({"image_id": image_id})
                else:
                    await dataset_collection.delete_many({"gridfs_id": image_id})
                
                results["deleted"].append(image_id)
                logger.info(f"🗑️  Bulk deleted image {image_id}")
            else:
                results["failed"].append({"id": image_id, "reason": "Image not found"})
                
        except Exception as e:
            results["failed"].append({"id": image_id, "reason": str(e)})
            logger.error(f"❌ Failed to delete image {image_id}: {e}")
    
    return {
        "status": "completed",
        "results": results,
        "bucket_type": bucket_type
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)