# CNN Image Classifier API with MongoDB Storage

A complete machine learning solution for image classification using Convolutional Neural Networks (CNN) with FastAPI backend and MongoDB storage system.

## 🚀 Features

- **CNN Model Training**: Train custom image classification models using TensorFlow/Keras
- **FastAPI REST API**: High-performance API for image predictions
- **MongoDB Integration**: Store images, predictions, and training datasets
- **GridFS Storage**: Efficient storage for large image files
- **Real-time Metrics**: Performance monitoring and system metrics
- **Async Operations**: Full async support for database operations
- **CORS Support**: Ready for frontend integration
- **Dataset Management**: Tools for dataset cleaning and management

## 📋 Requirements

- Python 3.11+ (recommended)
- MongoDB (local or cloud instance)
- TensorFlow 2.13+
- At least 4GB RAM for model training

## 🛠️ Installation

### Backend Setup

#### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd cnn-image-classifier
```

#### 2. Set Up Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Set Up MongoDB
Make sure MongoDB is running on your system:
```bash
# Local MongoDB
mongod

# Or use MongoDB Atlas (cloud)
# Update MONGODB_URL in the code files
```

### Frontend Setup

#### 1. Navigate to Frontend Directory
```bash
cd frontend
```

#### 2. Install Node.js Dependencies
```bash
npm install
# or
yarn install
```

#### 3. Configure API Endpoint
Update the API base URL in your frontend configuration to point to your backend:
```javascript
// In your frontend config file
const API_BASE_URL = "http://localhost:8000";  // Backend URL
```

## 📁 Project Structure

```
├── backend/              # Backend API files
│   ├── app.py           # FastAPI application
│   ├── train_model.py   # CNN model training script
│   ├── clean_dataset.py # Dataset cleaning utility
│   ├── requirements.txt # Python dependencies
│   ├── runtime.txt      # Python version for deployment
│   └── models/          # Trained models directory
│       ├── model.h5     # Trained CNN model
│       ├── class_names.json # Class labels
│       └── training_stats.json # Training statistics
├── frontend/             # React frontend application
│   ├── dist/            # Built frontend files
│   ├── node_modules/    # Node.js dependencies
│   ├── public/          # Static assets
│   ├── src/             # React source code
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── utils/       # Utility functions
│   │   ├── App.jsx      # Main App component
│   │   └── main.jsx     # Entry point
│   ├── package.json     # Node.js dependencies
│   ├── package-lock.json # Dependency lock file
│   ├── vite.config.js   # Vite configuration
│   ├── eslint.config.js # ESLint configuration
│   └── index.html       # HTML template
├── dataset/              # Training dataset
│   ├── cats/            # Cat images
│   ├── dogs/            # Dog images
│   └── ...              # Other classes
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## 🎯 Usage

### 1. Prepare Your Dataset

Organize your images in the following structure:
```
dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── dogs/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

### 2. Clean Dataset (Optional)
Remove corrupted images:
```bash
python clean_dataset.py
```

### 3. Train the Model
```bash
cd backend
python train_model.py
```

This will:
- Connect to MongoDB
- Load images from the database
- Train a CNN model
- Save the trained model to `models/model.h5`
- Save class names and training statistics

### 4. Start the Backend API Server
```bash
cd backend
python app.py
# Or using uvicorn directly:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 5. Start the Frontend Development Server
```bash
cd frontend
npm run dev
# or
yarn dev
```

The frontend will be available at `http://localhost:5173` (Vite default) or `http://localhost:3000`

### 6. Access the Application
- **Frontend**: `http://localhost:5173`
- **Backend API**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`

## 📖 API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Main Endpoints

#### 🏠 Health Check
```http
GET /
GET /health/
```

#### 🔮 Image Prediction
```http
POST /predict/
Content-Type: multipart/form-data

Body: file (image file - PNG, JPG, JPEG)
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "class": "cat",
    "confidence": 0.95,
    "processing_time": 0.123,
    "inference_time": 0.045,
    "file_size": 1024,
    "filename": "image.jpg",
    "image_id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "raw_predictions": [0.95, 0.05],
    "model_info": {
      "input_shape": "(None, 150, 150, 3)",
      "classes": ["cat", "dog"],
      "model_path": "models/model.h5"
    }
  }
}
```

#### 📊 Performance Metrics
```http
GET /metrics/
```

#### 🖼️ Stored Images
```http
GET /images/
GET /images/{image_id}
```

#### 📈 Prediction Statistics
```http
GET /predictions/stats/
```

#### 🧪 Model Testing
```http
POST /test-model/
```

## ⚙️ Configuration

### Backend Configuration

#### MongoDB Settings
Update these variables in your code files:

```python
# MongoDB Configuration
MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "image_classifier"
COLLECTION_NAME = "dataset_images"
BUCKET_NAME = "dataset_images"
```

#### Model Settings
```python
# Model Configuration
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
CLASS_NAMES = ["cat", "dog"]  # Update based on your classes
```

#### CORS Settings
```python
# Allowed origins for frontend
allow_origins=[
    "http://localhost:5173",  # Vite
    "http://localhost:3000",  # React
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "https://your-frontend-domain.com"  # Production frontend
]
```

### Frontend Configuration

#### Environment Variables
Create `.env` file in the frontend directory:
```bash
# Frontend environment variables
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_TITLE=CNN Image Classifier
VITE_MAX_FILE_SIZE=5242880  # 5MB in bytes
VITE_ALLOWED_FILE_TYPES=image/jpeg,image/jpg,image/png
```

#### Vite Configuration
Update `vite.config.js`:
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})
```

#### Package.json Scripts
Ensure your `package.json` has these scripts:
```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext js,jsx --report-unused-disable-directives --max-warnings 0"
  }
}
```

## 🚀 Deployment

### Frontend Deployment (Netlify/Vercel)

#### Build the Frontend
```bash
cd frontend
npm run build
# or
yarn build
```

#### Deploy to Netlify
1. Connect your GitHub repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `dist`
4. Add environment variable: `VITE_API_URL=your-backend-url`

#### Deploy to Vercel
1. Connect your GitHub repository to Vercel
2. Framework preset: Vite
3. Build command: `npm run build`
4. Output directory: `dist`
5. Add environment variable: `VITE_API_URL=your-backend-url`

### Backend Deployment (Render.com)

#### 1. Create Required Files
Create `runtime.txt` in the backend directory:
```
python-3.11.7
```

#### 2. Update CORS Settings
Make sure your `app.py` includes your frontend URL in CORS origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-domain.netlify.app",  # Add your frontend URL
        "https://your-frontend-domain.vercel.app",   # Or Vercel URL
        "http://localhost:5173",  # Keep for local development
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
)
```

#### 3. Deploy to Render
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set root directory: `backend`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   ```
   MONGODB_URL=your_mongodb_connection_string
   ```

### Full Stack Deployment with Docker

#### Frontend Dockerfile
```dockerfile
# frontend/Dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Backend Dockerfile
```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URL=mongodb://mongo:27017
    depends_on:
      - mongo

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

  mongo:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

## 🧪 Testing

### Backend Testing

#### Test Model Loading
```bash
curl -X POST "http://localhost:8000/test-model/"
```

#### Test Image Prediction
```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Frontend Testing

#### Run Frontend Tests
```bash
cd frontend
npm test
# or
yarn test
```

#### Frontend Development Commands
```bash
# Start development server
npm run dev

# Build for production  
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Full Stack Testing

#### Test Complete Flow
1. Start backend: `cd backend && python app.py`
2. Start frontend: `cd frontend && npm run dev`
3. Open browser: `http://localhost:5173`
4. Upload an image and verify prediction results

## 📊 Monitoring

The system provides comprehensive monitoring:

- **Request metrics**: Response times, success rates
- **Model performance**: Inference times, prediction accuracy
- **Database metrics**: Write times, storage usage
- **System metrics**: CPU, memory, disk usage
- **Image storage**: GridFS storage statistics

## 🔧 Troubleshooting

### Common Issues

1. **TensorFlow Installation Issues**
   ```bash
   pip install tensorflow-cpu==2.13.1
   ```

2. **MongoDB Connection Issues**
   - Ensure MongoDB is running
   - Check connection string
   - Verify network connectivity

3. **Model Loading Issues**
   - Ensure model file exists in `models/model.h5`
   - Check file permissions
   - Verify model compatibility

4. **Image Processing Issues**
   - Supported formats: PNG, JPG, JPEG
   - Check image file corruption
   - Verify file size limits

### Debug Mode
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- FastAPI for the high-performance web framework
- MongoDB for database storage
- Render.com for deployment platform

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation at `/docs`

---

**Built with ❤️ using TensorFlow, FastAPI, and MongoDB**
