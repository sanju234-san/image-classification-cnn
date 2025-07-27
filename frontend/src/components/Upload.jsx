import React, { useState, useRef } from 'react';

const Upload = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [previewUrl, setPreviewUrl] = useState(null);
    const [debugInfo, setDebugInfo] = useState('');
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            console.log('File selected:', file.name, file.type, file.size);
            
            // Validate file type
            const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
            if (!validTypes.includes(file.type)) {
                setError('Please select a valid image file (PNG, JPG, or JPEG).');
                return;
            }

            // Validate file size (e.g., max 10MB)
            const maxSize = 10 * 1024 * 1024; // 10MB
            if (file.size > maxSize) {
                setError('File size must be less than 10MB.');
                return;
            }

            setSelectedFile(file);
            setPrediction(null);
            setError('');
            setDebugInfo('');

            // Create preview URL
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
        }
    };

    const handleUpload = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        console.log('🚀 Upload button clicked');
        
        if (!selectedFile) {
            setError('Please select a file first.');
            return;
        }

        setLoading(true);
        setError('');
        setDebugInfo('🚀 Starting upload process...');

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            console.log('Uploading file:', selectedFile.name);
            console.log('FormData created:', formData.get('file'));
            setDebugInfo('📤 Uploading file to backend...');

            const API_URL = 'http://127.0.0.1:8000/predict/';
            console.log('Making request to:', API_URL);

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
                signal: controller.signal,
                headers: {
                    // Don't set Content-Type for FormData - let browser set it with boundary
                }
            });

            clearTimeout(timeoutId);
            console.log('Response status:', response.status);
            console.log('Response headers:', [...response.headers.entries()]);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', response.status, errorText);
                throw new Error(`Server error: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('✅ Full response data:', JSON.stringify(data, null, 2));
            setDebugInfo(`✅ Response received: ${JSON.stringify(data, null, 2)}`);

            // Parse the response based on your backend structure
            if (data && data.status === 'success' && data.prediction) {
                console.log('✅ Setting prediction:', data.prediction);
                setPrediction({
                    class: data.prediction.class,
                    confidence: data.prediction.confidence,
                    processingTime: data.prediction.processing_time,
                    inferenceTime: data.prediction.inference_time,
                    fileSize: data.prediction.file_size,
                    filename: data.prediction.filename
                });
                setDebugInfo('✅ Prediction set successfully!');
                setError(''); // Clear any previous errors
            } else {
                console.error('❌ Unexpected response format:', data);
                setDebugInfo(`❌ Unexpected response format: ${JSON.stringify(data)}`);
                setError('Unexpected response format from server');
            }

        } catch (err) {
            console.error('❌ Upload error:', err);
            if (err.name === 'AbortError') {
                const errorMsg = 'Request timeout - server took too long to respond';
                setError(errorMsg);
                setDebugInfo(`⏰ ${errorMsg}`);
            } else if (err.message.includes('Failed to fetch')) {
                const errorMsg = 'Cannot connect to backend. Make sure it\'s running on port 8000.';
                setError(errorMsg);
                setDebugInfo(`🔌 ${errorMsg}`);
            } else {
                const errorMsg = err.message;
                setError(errorMsg);
                setDebugInfo(`❌ ${errorMsg}`);
            }
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        setSelectedFile(null);
        setPrediction(null);
        setError('');
        setDebugInfo('');
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            setPreviewUrl(null);
        }
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const testBackendConnection = async () => {
        try {
            setDebugInfo('🔍 Testing backend connection...');
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000);
            
            const response = await fetch('http://127.0.0.1:8000/', {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            const data = await response.json();
            console.log('🏥 Backend status:', data);
            setDebugInfo(`✅ Backend connected!\nModel loaded: ${data.model_loaded}\nTotal requests: ${data.performance.total_requests}\nMongoDB: ${data.mongodb_connected}`);
            
            if (data.model_loaded) {
                setError('');
            } else {
                setError('⚠️ Model not loaded in backend');
            }
        } catch (err) {
            console.error('🚨 Backend connection test failed:', err);
            if (err.name === 'AbortError') {
                setDebugInfo('❌ Backend connection timeout');
                setError('Backend connection timeout. Make sure backend is running.');
            } else {
                setDebugInfo(`❌ Backend connection failed: ${err.message}`);
                setError('Cannot connect to backend. Make sure it\'s running on port 8000.');
            }
        }
    };

    return (
        <section className='flex items-center justify-center min-w-full min-h-[65vh] rounded-2xl bg-[url("https://lh3.googleusercontent.com/aida-public/AB6AXuD4FVfU7ff6z-2y6afKOonujol46LQBHvVqUixDqPj0DKoWvuI90_S0oHTU5HhSrvNY4efWCz33GXjXg7azKI-0w0QSPGgE6IgJVLqWdkg7Uo--WjWrLZs09gthMWaTivGnFCDa3w3ZqdxIQXAoTscIOZWou_6orB3O1zdzH2_-CKX9BMYRvuTPfJmRQ8yHixbOIRkCAhIIGm2vtvCzLJvJFfj7tF3kl4bZpBmT-HizJ70yxfLXCRQX8DEyqWaKL12kk4QLcqgUwnt4")] bg-cover bg-center h-64 w-full bg-no-repeat mb-15'>
            <div className="flex items-center justify-center min-w-full min-h-[65vh] bg-black/25 flex-col gap-6">
                <div className="flex flex-col items-center justify-center gap-3">
                    <h1 className="text-4xl font-bold text-white">Classify Your Images</h1>
                    <p className="text-xl font-medium text-white">
                        Upload an image to classify using our advanced AI model.
                    </p>
                </div>

                {/* Debug Panel */}
                <div className="w-full max-w-md">
                    <div className="flex gap-2 mb-2">
                        <button
                            onClick={testBackendConnection}
                            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors text-sm"
                        >
                            Test Backend
                        </button>
                        <button
                            onClick={() => {
                                setDebugInfo('');
                                setError('');
                                setPrediction(null);
                            }}
                            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors text-sm"
                        >
                            Clear Debug
                        </button>
                    </div>
                    
                    {debugInfo && (
                        <div className="bg-gray-800/90 text-white p-3 rounded text-xs mb-4 max-h-40 overflow-y-auto">
                            <div className="flex justify-between items-center mb-2">
                                <strong>🔍 Debug Info:</strong>
                                <button 
                                    onClick={() => setDebugInfo('')}
                                    className="text-gray-300 hover:text-white"
                                >
                                    ✕
                                </button>
                            </div>
                            <pre className="whitespace-pre-wrap">{debugInfo}</pre>
                        </div>
                    )}
                </div>

                {/* File Input */}
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={handleFileChange}
                    className="hidden"
                />

                {/* File Selection Area */}
                {!selectedFile ? (
                    <div 
                        onClick={() => fileInputRef.current?.click()}
                        className="border-2 border-dashed border-white/50 rounded-lg p-8 cursor-pointer hover:border-white/70 transition-colors"
                    >
                        <div className="text-center text-white">
                            <p className="text-lg mb-2">Click to select an image</p>
                            <p className="text-sm opacity-75">Supports PNG, JPG, JPEG (max 10MB)</p>
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col items-center gap-4">
                        {/* Image Preview */}
                        {previewUrl && (
                            <div className="max-w-xs max-h-64 overflow-hidden rounded-lg border-2 border-white/20">
                                <img 
                                    src={previewUrl} 
                                    alt="Preview" 
                                    className="w-full h-full object-cover"
                                />
                            </div>
                        )}
                        
                        {/* File Info */}
                        <div className="text-center text-white">
                            <p className="font-medium">{selectedFile.name}</p>
                            <p className="text-sm opacity-75">
                                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex gap-3">
                            <button
                                type="button"
                                onClick={handleUpload}
                                disabled={loading}
                                className="px-6 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {loading ? "Processing..." : "🚀 Classify Image"}
                            </button>
                            <button
                                type="button"
                                onClick={handleClear}
                                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
                                disabled={loading}
                            >
                                Clear
                            </button>
                        </div>
                    </div>
                )}

                {/* Loading State */}
                {loading && (
                    <div className="flex items-center gap-2 text-white">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        <p>Processing your image...</p>
                    </div>
                )}

                {/* Error Display */}
                {error && (
                    <div className="bg-red-500/80 text-white p-4 rounded-lg max-w-md text-center">
                        <p className="font-medium">❌ Error</p>
                        <p className="text-sm">{error}</p>
                    </div>
                )}

                {/* Prediction Results */}
                {prediction && (
                    <div className="bg-white/95 text-black p-6 rounded-lg shadow-lg max-w-md w-full">
                        <h2 className="text-xl font-bold mb-4 text-center text-green-700">🎯 Prediction Result</h2>
                        <div className="space-y-3">
                            <div className="flex justify-between items-center">
                                <span className="font-medium">Class:</span>
                                <span className="capitalize font-bold text-blue-600 text-lg">
                                    {prediction.class}
                                </span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="font-medium">Confidence:</span>
                                <span className="font-bold text-green-600 text-lg">
                                    {(prediction.confidence * 100).toFixed(2)}%
                                </span>
                            </div>
                            {prediction.processingTime && (
                                <div className="flex justify-between items-center">
                                    <span className="font-medium">Processing Time:</span>
                                    <span className="text-gray-600">
                                        {(prediction.processingTime * 1000).toFixed(0)}ms
                                    </span>
                                </div>
                            )}
                            {prediction.inferenceTime && (
                                <div className="flex justify-between items-center">
                                    <span className="font-medium">Model Inference:</span>
                                    <span className="text-gray-600">
                                        {(prediction.inferenceTime * 1000).toFixed(0)}ms
                                    </span>
                                </div>
                            )}
                            {prediction.fileSize && (
                                <div className="flex justify-between items-center">
                                    <span className="font-medium">File Size:</span>
                                    <span className="text-gray-600">
                                        {(prediction.fileSize / 1024).toFixed(1)} KB
                                    </span>
                                </div>
                            )}
                        </div>
                        
                        {/* Confidence Bar */}
                        <div className="mt-4">
                            <div className="flex justify-between text-sm mb-1">
                                <span>Confidence Level</span>
                                <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div className="bg-gray-200 rounded-full h-3">
                                <div 
                                    className={`h-3 rounded-full transition-all duration-1000 ${
                                        prediction.confidence > 0.8 ? 'bg-green-500' :
                                        prediction.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                                    }`}
                                    style={{ width: `${Math.max(prediction.confidence * 100, 5)}%` }}
                                ></div>
                            </div>
                        </div>

                        {/* Prediction Quality Indicator */}
                        <div className="mt-3 text-center">
                            {prediction.confidence > 0.8 && (
                                <span className="text-green-600 font-medium">🎯 High Confidence</span>
                            )}
                            {prediction.confidence > 0.6 && prediction.confidence <= 0.8 && (
                                <span className="text-yellow-600 font-medium">⚠️ Medium Confidence</span>
                            )}
                            {prediction.confidence <= 0.6 && (
                                <span className="text-red-600 font-medium">❓ Low Confidence</span>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </section>
    );
};

export default Upload;