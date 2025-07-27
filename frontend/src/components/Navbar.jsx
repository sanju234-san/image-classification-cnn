import React from 'react'
import Button from './Button'


const Navbar = () => {
    return (
        <nav className='flex items-center justify-between p-4 border-b-2 border-white px-15'>
            <div>
                <h1 className='text-2xl font-bold'>Image Classifier</h1>
            </div>

            <div className='flex gap-5'>
                <ul className='flex gap-5 items-center justify-center'>
                    <li className='text-lg font-semibold'>Home</li>
                    <li className='text-lg font-semibold'>About</li>
                    <li className='text-lg font-semibold'>Contact</li>
                    <Button btn='Upload Image'/>
                </ul>
            </div>
        </nav>
    )
}

export default Navbar
