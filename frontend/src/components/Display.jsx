import React from 'react'
import Card from './Card'

const Display = () => {
    return (
        <section className="">
            <h1 className="text-3xl font-bold mb-10">
                Example Classification
            </h1>
            <div className='flex gap-5 mb-15'>
                <Card
                    image='https://lh3.googleusercontent.com/aida-public/AB6AXuCiO2msnP3Q6w8memrR4Jqoxk6dgb_9Zjqb4bHcmDA4PO1JWTot_rDmNOyVdhVchwv1PzSThX0Q17gAi9YNmbNDP5Tvf5s6g5xjFGR-BW97cptKY9joQK8M4ucefXA1xLtPGcycMAlK1Y_FzUd1lMFIVfaeTRqosRLu2sPcChJlctTfZNKI6AwJWCYdSzkcklLKDMSOdLvuye8E_k-EFpCu9yr_mE4N8nagmBywfSs5fz1Fig0zkbAnNjv4ohdpk7A0qGGvs83NiQ-I '
                    name='Urban Landscape'
                    desc='A serene landscape with
mountains and a lake.'
                />
                <Card
                    name='Wildlife'
                    desc='A bustling city with skyscrapers
and traffic.'
                    image='https://lh3.googleusercontent.com/aida-public/AB6AXuCxOiG21M3Dw8AjUBczLJ_p6ov93m-DOaAxp-awneEGbPheOzWVUjpea6MyPYokGVCZFJ38T62x6j1KU6s9fn4JsAvIIOxQdI9j6TSX_JH6PEd8G1kybbnuVwBwZC_9jaaVV_zbzyTvD6AgYFhM0PU3hZCoiyCVoAo2PNXzVqNBXdBpxVBixWoiTktFp51fg_Y3QIpWoLlxcnRCxShaq6g_YDoe9IQAAS_t65-Vphhi0Ss4omb-a5EYsqVseQA8XPtxjlSlJMF6vXeL '
                />
                <Card
                    name='Abstract Art'
                    desc='A majestic lion in its natural
habitat.'
                    image='https://lh3.googleusercontent.com/aida-public/AB6AXuBAKQWCYD5dCwzDuIV0Mx0DsOx5f3sYpuWQ9JSfNyQlBvKsuTDq2653PbGqejhKeliSVUkaxqagnNOafoocRyGtrVBURVWbEoqDxpo5_MdIW86Vw4VcrecsEVDFYtVLB3eD6eJ1fiZ5pOzEC1aUGxBhOEYIxwQcvGqL0uy_l7Pia845poRaAzfxEKJKSEPj1nPcqo7WITwRcpBU-WYzkwKNZFg49VjRogG5tyxkID4MSBobZ2669cqWioGNOkSL1WQaVXhLIqvA4vaB'
                />
                <Card
                    name='Nature Scene'
                    desc='A colorful and dynamic abstract
painting.'
                    image='https://lh3.googleusercontent.com/aida-public/AB6AXuAMjZn2LFfl0jEuYJu0lR_DR1V_HyMQ7uMC-JKSqoOKVYQwD9iUL7wjNE2eTwosLHoSHX35Mj49dQTtNjYr0SvmRI63xoY0bp9cXQpHku2V7VgD7kz6F82e91DCWU7wYGWadZbXhdNvtvGFNx3SSgvsiTUGkPJ2yBxcqzWieyIFRvCl7rfxhTeVNExwFLtp7CBZ2UWDDr5H4QlWcFIZTkaIXdSUJ2Ues4ZivmWPv3TG08sawY5zp9gZMK_XqtW44CN6CxL9UymFL8Z9'
                />
                
            </div>
        </section>
    );
}

export default Display
