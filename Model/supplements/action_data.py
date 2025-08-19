# action_data.py

actions = [
    ('walking', 'walking in a virtual environment'),
    ('running', 'running in a virtual environment'),
    ('jumping', 'jumping in a virtual environment'),
    ('bending down', 'bending down in a virtual environment'),
    ('stand', 'standing in a virtual environment'),
    ('squatting', 'squatting in a virtual environment'),
    ('raising hand', 'raising hand in a virtual environment'),
    ('shooting', 'holding a virtual gun and shooting at the virtual boss'),
    ('waive', 'waiving a laser sword or hammer to fight against the boss'),
    ('throw', 'throw a virtual hammer to the boss in the virtual environment'),
    ('cut', 'using a laser sword to hit back coming objects'),
    ('bowling', 'catch a virtual bowling ball and throw it to the target'),
    ('move using controller', 'moving using controller in the virtual environment'),
    ('waive sword', 'waive sword to hit the coming virtual objects'),
    ('measure length', 'selecting two points in the virtual environment with controller to measure the length'),
    ('picking up an item from the table', 'picking up an item from the table'),
    ('throwing a net to catch fish', 'throw a virtual net into the water to catch fish'),
    ('grab and collect box', 'grab the box in the virtual environment with controller and collect them')
]

motion_type_dict = {
    'walking': 'cyclic',
    'running': 'cyclic',
    'jumping': 'explosive',
    'bending down': 'static',
    'stand': 'static',
    'squatting': 'static',
    'raising hand': 'static',
    'shooting': 'dynamic',
    'waive': 'cyclic',
    'throw': 'dynamic',
    'cut': 'dynamic',
    'bowling': 'dynamic',
    'move using controller': 'cyclic',
    'waive sword': 'cyclic',
    'measure length': 'static',
    'picking up an item from the table': 'dynamic',
    'throwing a net to catch fish': 'dynamic',
    'grab and collect box': 'dynamic'
}

jumping_dict = {k: ('yes' if k == 'jumping' else 'no') for k, _ in actions}

hand_usage_dict = {
    'walking': 'none',
    'running': 'none',
    'jumping': 'none',
    'bending down': 'none',
    'stand': 'none',
    'squatting': 'none',
    'raising hand': 'one',
    'shooting': 'two',
    'waive': 'two',
    'throw': 'one',
    'cut': 'two',
    'bowling': 'two',
    'move using controller': 'two',
    'waive sword': 'one',
    'measure length': 'two',
    'picking up an item from the table': 'two',
    'throwing a net to catch fish': 'two',
    'grab and collect box': 'two'
}

limbs_involved_dict = {
    'walking': 'legs only',
    'running': 'legs only',
    'jumping': 'full body',
    'bending down': 'torso and legs',
    'stand': 'full body',
    'squatting': 'legs only',
    'raising hand': 'arms only',
    'shooting': 'arms and torso',
    'waive': 'arms only',
    'throw': 'arms and torso',
    'cut': 'arms only',
    'bowling': 'arms and legs',
    'move using controller': 'arms and torso',
    'waive sword': 'arms only',
    'measure length': 'arms only',
    'picking up an item from the table': 'arms and torso',
    'throwing a net to catch fish': 'arms and torso',
    'grab and collect box': 'arms and torso'
}
