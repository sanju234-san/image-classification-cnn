import React from 'react'
import Upload from '../components/Upload'
import Display from '../components/Display'
import Button from '../components/Button'

const Home = () => {
    return (
        <div className='mt-15 w-[70%] mx-auto min-h-screen'>
            <Upload />

            <div className='flex flex-col w-full h-auto gap-3 mb-15'>
                <h1 className='text-3xl font-bold'>
                    How It Works
                </h1>

                <p className='text-xl'>
                    Our image classifier uses a state-of-the-art deep learning model to analyze and categorize images. Simply upload an
                    image, and our system will identify the objects, scenes, or concepts present in the image with high accuracy. The
                    model has been trained on a vast dataset of diverse images to ensure robust performance across various domains.
                </p>
            </div>

            <Display />

            <section className='flex flex-col items-center justify-center w-full h-40 mt-25'>
                <h1 className='self-start text-3xl font-bold'>
                    Get Started
                </h1>
                <div className='my-15'>
                    <Button btn='Upload Image'/>
                </div>
            </section>
        </div>
    )
}

export default Home
