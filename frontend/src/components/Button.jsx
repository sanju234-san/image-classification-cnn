import React, { useRef } from 'react'
import axios from 'axios'

const Button = (props) => {
    const fileInputRef = useRef(null)

    const handleClick = () => {
        // Trigger the hidden file input
        fileInputRef.current.click()
    }

    const handleFileChange = async (event) => {
        const file = event.target.files[0]

        if (!file) {
            alert('No file selected')
            return
        }

        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file')
            return
        }

        // Create FormData to send the image
        const formData = new FormData()
        formData.append('image', file)

        try {
            // Show loading state
            alert('Uploading image...')

            // Send to backend using axios - replace with your actual endpoint
            const response = await axios.post('/api/upload-image', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                // Optional: Track upload progress
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
                    console.log(`Upload Progress: ${percentCompleted}%`)
                }
            })

            // Axios automatically parses JSON response
            alert('Image processed successfully!')
            console.log('Backend response:', response.data)

            // Call optional callback function with result
            if (props.onImageProcessed) {
                props.onImageProcessed(response.data)
            }

        } catch (error) {
            console.error('Error uploading image:', error)

            // Handle different types of axios errors
            if (error.response) {
                // Server responded with error status
                alert(`Server error: ${error.response.status} - ${error.response.data.message || 'Upload failed'}`)
            } else if (error.request) {
                // Request was made but no response received
                alert('No response from server. Please check your connection.')
            } else {
                // Something else happened
                alert('Failed to process image. Please try again.')
            }
        }

        // Reset the input
        event.target.value = ''
    }

    return (
        <>
            <button
                className='w-35 h-full bg-amber-600 p-3 rounded-lg text-black font-bold cursor-pointer hover:bg-amber-700 transition-colors'
                onClick={handleClick}
            >
                {props.btn || 'Upload Image'}
            </button>

            {/* Hidden file input */}
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                style={{ display: 'none' }}
            />
        </>
    )
}

export default Button