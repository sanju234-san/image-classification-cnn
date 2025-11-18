# CNN Image Classifier API

A full-stack image classification application using Convolutional Neural Networks (CNN) to classify images of cats and dogs. Built with FastAPI backend and React frontend, with MongoDB for data storage.

## 🚀 Features

- **Real-time Image Classification**: Upload images and get instant predictions
- **CNN Model**: Custom-trained TensorFlow/Keras model for cat vs dog classification
- **MongoDB Integration**: Store predictions, images, and datasets using MongoDB with GridFS
- **Performance Monitoring**: Track system metrics, inference times, and request statistics
- **RESTful API**: Clean and documented API endpoints
- **Modern UI**: React-based frontend with Tailwind CSS
- **CORS Support**: Cross-origin resource sharing enabled for frontend integration

## 📋 Prerequisites

- Python 3.11.7
- Node.js (for frontend)
- MongoDB (local installation or MongoDB Compass)
- pip (Python package manager)
- npm or yarn (Node package manager)

## 🛠️ Installation

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <project-directory>
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MongoDB**
   - Install MongoDB locally or use MongoDB Compass
   - Default connection: `mongodb://localhost:27017/`
   - Database name: `image_classifier`

5. **Configure environment variables**
   - Copy `.env` file and update if needed
   - Key settings:
     - `MONGODB_CONNECTION_STRING`: MongoDB connection URL
     - `MODEL_PATH`: Path to your trained model (default: `models/model.h5`)
     - `APP_PORT`: API port (default: 8000)

6. **Train the model** (if you don't have a pre-trained model)
   ```bash
   python train_model.py
   ```
   This will:
   - Load training data from MongoDB
   - Train a CNN model
   - Save the model to `models/model.h5`
   - Save class names and training statistics

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend  # Adjust path as needed
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure API endpoint**
   - Update the API URL in your frontend code if needed
   - Default: `http://localhost:8000`

## 🚀 Running the Application

### Start Backend Server

```bash
# Make sure you're in the backend directory and venv is activated
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

### Start Frontend Development Server

```bash
# In the frontend directory
npm run dev
```

The frontend will be available at: `http://localhost:5173` (or the port shown in terminal)

## 📚 API Documentation

Once the backend is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Main Endpoints

#### `GET /`
Returns API status and configuration

**Response:**
```json
{
  "message": "Welcome to CNN Image Classifier API",
  "status": "healthy",
  "model_loaded": true,
  "database_info": {...},
  "performance": {...}
}
```

#### `POST /predict/`
Upload an image for classification

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Form data with `file` field (image file)

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "class": "cat",
    "confidence": 0.98,
    "processing_time": 0.15,
    "inference_time": 0.08,
    "image_id": "..."
  },
  "database_stored": true
}
```

## 📁 Project Structure

```
.
├── app.py                  # Main FastAPI application
├── train_model.py          # Model training script
├── clean_dataset.py        # Dataset cleaning utility
├── requirements.txt        # Python dependencies
├── runtime.txt            # Python version specification
├── .env                   # Environment configuration
├── models/
│   └── model.h5           # Trained CNN model
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   ├── Pages/
│   │   │   └── Home.jsx   # Home page component
│   │   └── components/
│   │       └── Navbar.jsx # Navigation component
│   ├── index.css          # Tailwind CSS imports
│   └── package.json       # Frontend dependencies
└── README.md              # This file
```

## 🎯 Model Architecture

The CNN model consists of:
- **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Flatten Layer**
- **Dense Layer**: 128 neurons, ReLU activation
- **Output Layer**: Softmax activation (2 classes: cat, dog)

**Input Shape**: 150x150x3 (RGB images)

## 🗄️ Database Collections

The application uses the following MongoDB collections:

- **predictions**: Stores prediction results with metadata
- **uploaded_images**: Metadata for user-uploaded images
- **cats**: Dataset images for cat class
- **dogs**: Dataset images for dog class

**GridFS Buckets**:
- `images`: Stores user-uploaded images
- `dataset_images`: Stores training dataset images

## 🔧 Configuration

Key configuration options in `.env`:

```env
# Application
APP_PORT=8000
APP_DEBUG=False

# MongoDB
MONGODB_CONNECTION_STRING="mongodb://localhost:27017/"
MONGODB_DATABASE="image_classifier"

# Model
MODEL_PATH="models/model.h5"
MODEL_CLASSES="cat,dog"
MODEL_INPUT_HEIGHT=150
MODEL_INPUT_WIDTH=150

# File Upload
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS="png,jpg,jpeg,webp"
```

## 📊 Performance Monitoring

The API tracks various metrics:
- Request processing times
- Model inference times
- Database write times
- Image storage times
- System resource usage (CPU, memory, disk)

Access metrics at the root endpoint (`/`)

## 🧹 Dataset Cleaning

Before training, clean your dataset:

```bash
python clean_dataset.py
```

This removes corrupted images that can't be opened by PIL.

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `models/model.h5` exists
- Check file permissions
- Verify TensorFlow/Keras compatibility

### MongoDB Connection Failed
- Verify MongoDB is running: `mongod --version`
- Check connection string in `.env`
- Ensure correct database permissions

### CORS Errors
- Frontend origin is included in CORS settings
- Check `allow_origins` in `app.py`

### Image Upload Fails
- Check file size (max 10MB by default)
- Verify file format (PNG, JPG, JPEG, WEBP)
- Ensure image is valid (not corrupted)

## 📝 Development

### Adding New Classes

1. Update `CLASS_NAMES` in `app.py`
2. Update `MODEL_CLASSES` in `.env`
3. Retrain model with new dataset
4. Update frontend UI to handle new classes

### Training with Custom Dataset

1. Upload images to MongoDB collections (one collection per class)
2. Update collection names in `train_model.py`
3. Run training script
4. Update class names in configuration

## 📄 License

[Add your license here]

## 👥 Contributing

[Add contributing guidelines here]

## 📧 Contact

[Add your contact information here]

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- FastAPI for the web framework
- MongoDB for database storage
- React for frontend framework
