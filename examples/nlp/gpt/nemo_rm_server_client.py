from utils.rpc import RPCClient


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

SYSTEM_PROMPT_TEMPLATE = "<extra_id_0>System\n{value}\n"

USER_TURN_TEMPLATE = "<extra_id_1>User\n{value}\n"

ASSISTANT_TURN_TEMPLATE = "<extra_id_1>Assistant\n{value}\n"

LABEL_PREFIX = "<extra_id_2>"

def dataloader_to_template(data):
    """
    Input in batches:
    [
        {'content': 
            [
                'How do I detail a car?', 
                'Who created the Superman cartoon character?'
            ], 
            'role': 
            [
                'user', 
                'user'
            ]
        }, 
        {'content': 
            [
                "Detailing a car involves a thorough cleaning inside and out, as well as polishing and waxing to protect the vehicle's surfaces. Here's a step-by-step guide to detailing a car:\n\n**Exterior Detailing:**\n\n1. **Wash the Car:**\n   - Rinse the car with water to remove loose dirt.\n   - Use a car wash soap and microfiber wash mitt to clean the car from top to bottom.\n   - Clean the wheels and tires with a brush and a wheel cleaner.\n   - Rinse the car thoroughly to remove all soap.\n\n2. **Dry the Car:**\n   - Use a microfiber towel or a chamois to dry the car to prevent water spots.\n\n3. **Clay Bar Treatment:**\n   - Use a clay bar with a lubricant to remove embedded surface contaminants from the paint.\n\n4. **Polishing:**\n   - Apply car polish with a dual-action polisher or by hand to correct paint imperfections and create a smooth surface.\n\n5. **Waxing:**\n   - Apply a coat of wax or paint sealant to protect the paint and give it a glossy finish.\n\n6. **Windows and Mirrors:**\n   - Clean the windows and mirrors with a glass cleaner and a microfiber towel.\n\n7. **Tire and Trim Dressing:**\n   - Apply a tire dressing to the tires for a shiny finish.\n   - Use a trim restorer or protectant on plastic and rubber parts to prevent fading.\n\n**Interior Detailing:**\n\n1. **Remove Trash:**\n   - Clear out any trash and remove personal items from the car.\n\n2. **Vacuum:**\n   - Vacuum the seats, carpets, floor mats, and trunk.\n   - Use a brush attachment for the dashboard and door panels.\n\n3. **Shampoo Carpets and Upholstery:**\n   - Use a carpet cleaner and a brush to clean the carpets and upholstery.\n   - For leather interiors, use a leather cleaner and conditioner.\n\n4. **Clean Hard Surfaces:**\n   - Wipe down all hard surfaces (dashboard, center console, door panels, etc.) with a mild all-purpose cleaner and a microfiber cloth.\n\n5. **Windows and Mirrors:**\n   - Clean the interior side of windows and mirrors.\n\n6. **Air Vents and Crevices:**\n   - Use a detailing brush or compressed air to clean out air vents and hard-to-reach crevices.\n\n7. **Final Touches:**\n   - Apply a protectant to the dashboard and other plastic components.\n   - Replace air fresheners if needed.\n\n**Additional Tips:**\n\n- Work in the shade or a cool, well-ventilated garage to prevent products from drying too quickly and leaving residue.\n- Use separate buckets for washing and rinsing to avoid contaminating the clean water with dirt.\n- Always use gentle, non-abrasive materials and cleaners specifically designed for automotive use to avoid damaging surfaces.\n- Move in a systematic way to ensure you don't miss any spots.\n\nBy following these steps, you'll give your car a thorough clean that not only makes it look great but also helps in maintaining its value. Remember, regular detailing can prevent wear and tear and keep your car looking new for years to come.", 
                "Superman, the iconic comic book superhero, was created by writer Jerry Siegel and artist Joe Shuster. Superman first appeared in Action Comics #1, which was published by Detective Comics, Inc. (later DC Comics) in June 1938. The character's immense popularity established him as one of the most enduring and recognizable figures in the superhero genre."
            ], 
            'role': [
                'assistant', 
                'assistant'
            ]
        }
    ]

    """

    assert len(data) == 2

    user_data = data[0]
    assistant_data = data[1]

    batch_size = len(user_data['content'])

    user_content = user_data['content']
    assistant_content = assistant_data['content']

    samples = []
    for i in range(0,batch_size):
        text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)
        text += USER_TURN_TEMPLATE.format(value=user_content[i])
        text += ASSISTANT_TURN_TEMPLATE.format(value=assistant_content[i])
        text += LABEL_PREFIX
        samples.append(text)

    return samples


server = RPCClient('0.0.0.0', 7777)
server.connect()


batch = [
    {'content': 
        [
            'How do I detail a car?', 
            'Who created the Superman cartoon character?'
        ], 
        'role': 
        [
            'user', 
            'user'
        ]
    }, 
    {'content': 
        [
            "Detailing a car involves a thorough cleaning inside and out, as well as polishing and waxing to protect the vehicle's surfaces. Here's a step-by-step guide to detailing a car:\n\n**Exterior Detailing:**\n\n1. **Wash the Car:**\n   - Rinse the car with water to remove loose dirt.\n   - Use a car wash soap and microfiber wash mitt to clean the car from top to bottom.\n   - Clean the wheels and tires with a brush and a wheel cleaner.\n   - Rinse the car thoroughly to remove all soap.\n\n2. **Dry the Car:**\n   - Use a microfiber towel or a chamois to dry the car to prevent water spots.\n\n3. **Clay Bar Treatment:**\n   - Use a clay bar with a lubricant to remove embedded surface contaminants from the paint.\n\n4. **Polishing:**\n   - Apply car polish with a dual-action polisher or by hand to correct paint imperfections and create a smooth surface.\n\n5. **Waxing:**\n   - Apply a coat of wax or paint sealant to protect the paint and give it a glossy finish.\n\n6. **Windows and Mirrors:**\n   - Clean the windows and mirrors with a glass cleaner and a microfiber towel.\n\n7. **Tire and Trim Dressing:**\n   - Apply a tire dressing to the tires for a shiny finish.\n   - Use a trim restorer or protectant on plastic and rubber parts to prevent fading.\n\n**Interior Detailing:**\n\n1. **Remove Trash:**\n   - Clear out any trash and remove personal items from the car.\n\n2. **Vacuum:**\n   - Vacuum the seats, carpets, floor mats, and trunk.\n   - Use a brush attachment for the dashboard and door panels.\n\n3. **Shampoo Carpets and Upholstery:**\n   - Use a carpet cleaner and a brush to clean the carpets and upholstery.\n   - For leather interiors, use a leather cleaner and conditioner.\n\n4. **Clean Hard Surfaces:**\n   - Wipe down all hard surfaces (dashboard, center console, door panels, etc.) with a mild all-purpose cleaner and a microfiber cloth.\n\n5. **Windows and Mirrors:**\n   - Clean the interior side of windows and mirrors.\n\n6. **Air Vents and Crevices:**\n   - Use a detailing brush or compressed air to clean out air vents and hard-to-reach crevices.\n\n7. **Final Touches:**\n   - Apply a protectant to the dashboard and other plastic components.\n   - Replace air fresheners if needed.\n\n**Additional Tips:**\n\n- Work in the shade or a cool, well-ventilated garage to prevent products from drying too quickly and leaving residue.\n- Use separate buckets for washing and rinsing to avoid contaminating the clean water with dirt.\n- Always use gentle, non-abrasive materials and cleaners specifically designed for automotive use to avoid damaging surfaces.\n- Move in a systematic way to ensure you don't miss any spots.\n\nBy following these steps, you'll give your car a thorough clean that not only makes it look great but also helps in maintaining its value. Remember, regular detailing can prevent wear and tear and keep your car looking new for years to come.", 
            "Superman, the iconic comic book superhero, was created by writer Jerry Siegel and artist Joe Shuster. Superman first appeared in Action Comics #1, which was published by Detective Comics, Inc. (later DC Comics) in June 1938. The character's immense popularity established him as one of the most enduring and recognizable figures in the superhero genre."
        ], 
        'role': [
            'assistant', 
            'assistant'
        ]
    }
]

samples = dataloader_to_template(batch)
results = server.get_rm_scores(samples)
print(results)

server.disconnect()