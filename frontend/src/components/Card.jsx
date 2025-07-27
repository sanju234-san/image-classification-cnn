import React from 'react'

const Card = (props) => {
    return (
        <div className="flex flex-col items-center">
            <img src={props.image} alt="image" className='rounded-xl'/>
            <div className='mt-4'>
                <h1 className='text-lg font-bold'>
                    {props.name}
                </h1>
                <p className='text-md text-[#e0cebc]'>
                    {props.desc}
                </p>
            </div>
        </div>
    )
}

export default Card
