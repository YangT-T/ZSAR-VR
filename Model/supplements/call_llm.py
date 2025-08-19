import os
import time
import pandas as pd
from dashscope import Generation

# 设置 DashScope API_KEY
DASHSCOPE_API_KEY = "sk-a5503555e4b64d8aab4e14c49ca1d214"  # 替换为你的 API Key

import os
from openai import OpenAI



# 输入动作列表
actions = [
    'walking: walking in a virtual environment',
    'running: running in a virtual environment',
    'jumping: jumping in a virtual environment',
    'bending down: bending down in a virtual environment',
    'stand: standing in a virtual environment',
    'squatting: squatting in a virtual environment',
    'raising hand: raising hand in a virtual environment',
    'shooting: holding a virtual gun and shooting at the virutal boss',
    'waive: waiving a laser sword or hammer to fight against the boss',
    'throw: thorw a virtual hammer to the boss in the virtual environment',
    'cut: using a laser sword to hit back comming objects',
    'bowling: catch a vritual bowling ball and throw it to the target',
    'move using controller: moving using controller in the virtual environment',
    'waive sword: waive sword to hit the comming virtual objects.',
    'measure length: selecting two points in the virtual environment with controller to measure the length',
    'picking up an item from the table',
    'throwing a net to catch fish: throw a virtual net into the water to catch fish',
    'grab and collect box: grab the box in the virtual environment with controller and collect them'
]

# prompt_template = """
# You are given the name of a human action: "{action}". Your task is to describe this action using the following 10 components, in a realistic and distinguishable way:

# - Head: Describe head orientation or movement based solely on body motion.
# - Hands: Focus on how arms and hands move, including coordination and rhythm.
# - Torso: Describe torso posture and movement only from the perspective of body kinematics, without referencing environment or objects.
# - Legs: Describe leg movement, joint involvement, and body support.
# - Start: Describe initial posture (head, torso, limbs) before action begins.
# - Middle: Describe dynamic motion in progress.
# - End: Describe body posture at the completion of action.
# - Global Desc: Give a 2-8 word whole-body summary of the action motion.
# - VR Scenario: Describe the virtual environment or user’s context.
# - Interaction: If the action is in ['walking', 'running', 'jumping', 'bending down', 'stand', 'squatting', 'raising hand'], write exactly "unrelated". Otherwise, describe interaction with virtual elements using neutral language (e.g., "reaches forward", "grabs", "moves").

# Formatting Rules:
# - Each line starts with the component name (e.g., "Head:") followed by a descriptive clause.
# - Each line should be no more than 8 words and uses natural English.
# - Avoid repeating sentence structures across components.
# - Do not mention any external object in Head, Torso, Legs unless explicitly stated in Interaction or Scenario.
# - Ensure all 10 lines are returned, in order.

# For example:

# Head: tilt back slightly  
# Hands: hold toothbrush and move it in circular motions
# Torso: remain still
# Legs: remain still
# Start: Pick up a toothbrush.  
# Middle: Scrub teeth.
# End: Rinse mouth.
# Global Desc:  pick up a toothbrush and scrub their teeth.
# VR Scenario: A person brushing in a virtual environment.  
# Interaction: unrelated

# Now generate the 10-line description for the action "{action}":
# """


# prompt_template2 = """
# You are given the name of a human action: "{action}". Your task is to describe this action using the following 10 components, focusing on *human skeletal motion only*.

# Your goal is to generate **accurate, concise, and discriminative** descriptions for each body part. If two actions share the same motion pattern in a body part (e.g., torso remains still), their descriptions **must be textually identical** for that part. If the motion is different, then use **distinct and specific** descriptions to help distinguish them.

# Descriptions should be **based purely on body movement**, not on external objects, intentions, or scene context (except in "VR Scenario" and "Interaction").

# Use the following structure:

# - Head: Orientation or motion (e.g., tilt, gaze direction).
# - Hands: Arm/hand movement (e.g., side, height, rhythm, symmetry).
# - Torso: Posture or movement of the torso/spine (e.g., upright, bent forward, twisted).
# - Legs: Whether legs are static or moving, weight shifts, stepping.
# - Start: Initial whole-body pose.
# - Middle: Ongoing or peak motion.
# - End: Final pose and limb positions.
# - Global Desc: A 3–8 word summary of full-body motion.
# - VR Scenario: Describe virtual context.
# - Interaction: If the action is in ['walking', 'running', 'jumping', 'bending down', 'stand', 'squatting', 'raising hand'], write exactly "unrelated". Otherwise, describe interaction with virtual objects.

# Formatting Rules:
# - Each line must start with the component name (e.g., "Head:") and be no more than 10 words.
# - Use consistent wording for identical motions across actions.
# - Use varied and specific language only when the motion differs.
# - Do not infer intent, object, or outcome unless for "VR Scenario" or "Interaction".
# - Avoid overly generic terms like "move" without qualifiers (e.g., "moves right arm upward" is good, "moves arm" is vague).
# - Ensure all 10 lines are returned, in order.

# For example:
# Head: tilt back slightly  
# Hands: hold toothbrush and move it in circular motions
# Torso: remain still
# Legs: remain still
# Start: Pick up a toothbrush.  
# Middle: Scrub teeth.
# End: Rinse mouth.
# Global Desc:  pick up a toothbrush and scrub their teeth.
# VR Scenario: A person brushing in a virtual environment.  
# Interaction: unrelated

# Now write the 10-line description for the action "{action}":
# """


# prompt_config.py

# prompt_template2 = """
# You are given the name of a human action: "{action}". Your task is to describe this action using the following 10 structured components, focusing **strictly on skeletal body motion only** — excluding any references to virtual objects, gaze targets, intentions, or scene semantics.

# Your descriptions must be:
# - **Accurate** (describe real motion patterns),
# - **Concise** (≤10 words per line),
# - **Discriminative** (distinct motions across actions),
# - **Consistent** (identical motions → identical wording across actions).

# You may use the following prior information to guide motion characteristics:
# - Motion Type: {motion_type}
# - Involves Jumping: {jumping}
# - Hand Usage: {hand_usage}
# - Limbs Involved: {limbs_involved}

# Use this structure (10 fields):

# 1. **Head**:  
#    Describe only the skeletal orientation and movement of the head (e.g., tilt, rotation, stillness).  
#    ❌ Do not include gaze targets, directions, or external references.  
#    ✅ Examples: "remains upright", "tilts forward slightly", "rotates side to side".

# 2. **Hands**:  
#    Describe arm and hand motion in terms of joint position and motion direction.  
#    Use distinctive phrasing like “raise above shoulder”, “extend overhead”, “remain alongside torso”.  
#    ✅ Examples: "raise one hand above head", "arms rest along body", "swing forward rhythmically"

# 3. **Torso**:  
#    Emphasize spine angle changes.  
#    ✅ Examples: "torso bends forward from lumbar", "remains vertically aligned", "leans backward slightly"


# 4. **Legs**:  
#    Focus on lower limb motion — stepping, squatting, jumping, static posture.  
#    ❌ Avoid direction or object-related phrasing (e.g., "step toward table").  
#    ✅ Examples: "alternate stepping motion", "bend at knees", "stand still".

# 5. **Start**:  
#    Describe the initial full-body pose (arms, legs, head).  
#    ❌ No objects like “holding ball” or context like “ready to shoot”.  
#    ✅ Examples: "feet together, arms at sides", "standing with knees slightly bent".

# 6. **Middle**:  
#    Describe peak motion (dynamic part of the action).  
#    ❌ Avoid intention/target verbs like “throws”, “hits”.  
#    ✅ Examples: "raise one hand overhead", "jump upward", "rotate torso while stepping".

# 7. **End**:  
#    Describe the final pose only, not the result.  
#    ❌ Avoid outcomes like “after throwing” or “after hitting object”.  
#    ✅ Examples: "arms lowered, knees slightly bent", "standing still".

# 8. **Global Desc**:  
#    Give a **short full-body motion summary** (3–8 words).  
#    ❌ Must not describe context, task, or object.  
#    ✅ Examples: "rhythmic forward stepping with arm swing", "squat and rise with trunk upright".

# 9. **VR Scenario**:  
#    Now describe how the action would appear in a virtual environment.  
#    ✅ This is the **only field allowed** to mention objects, environment, or goals.  
#    ✅ Example: "walking down a virtual corridor".

# 10. **Interaction**:  
#    If the action is in the basic list (below), write `"unrelated"` exactly.  
#    Otherwise, describe what virtual interaction (with objects or tasks) occurs.  
#    Basic (always "unrelated"):  
#    ['walking', 'running', 'jumping', 'bending down', 'stand', 'squatting', 'raising hand']

# ---

# **Formatting Rules**:
# - Return exactly 10 lines in the above order.
# - Each line starts with field name, e.g., "Head:", "Hands:".
# - Each line must be ≤10 words.
# - Use **identical phrasing** across actions for the same motion.
# - Use **distinct phrases** for different motions.
# - Avoid vague verbs like "move" — use e.g., "raise right arm" instead.

# Now write the 10-line description for the action "{action}":
# """



prompt_template2 = """
You are given the name of a human action: "{action}"  
And the full list of all action classes:  
{action_list}

Your task is to describe this action using the following **10 structured fields**, focusing entirely on **skeletal body motion**.

---

Each field should:
- Be based purely on the body's movement, posture, or gesture
- Avoid repetition of generic phrases like "remains upright" unless strictly accurate
- Use motion-specific verbs (e.g., tilt, extend, bend, raise)
- Be **maximally discriminative** compared to the other listed actions —  
  ⚠️ you must ensure each field’s wording is distinct from descriptions of other actions
- Be anatomically and physically plausible
- Use concise natural language (≤12 words per field)
- Avoid abstract goals, task intentions, or object names (except in VR Scenario / Interaction)

---

⚠️ Formatting Instructions:
- Output **exactly 10 lines**
- Each line must begin with a field name followed by a colon (e.g., "Head: ...")
- Do **not** include any numbering or bullet points
- Do **not** include comparisons like “unlike X” — instead, implicitly make each field unique
- Keep wording consistent for same motion across actions; otherwise, maximize specificity

---

Fields to fill:

Head  
Hands  
Torso  
Legs  
Start  
Middle  
End  
Global Desc  
VR Scenario  
Interaction

---

Field Guidelines:

Head:  
Describe the motion or orientation of the head only.  
Avoid references to gaze targets or intention.  
✅ Examples: "tilts slightly forward", "rotates side to side", "remains upright"

Hands:  
Describe hand or arm movement including pose, symmetry, and height.  
Avoid object/tool names unless essential.  
✅ Examples: "raise one hand above shoulder", "arms hang at sides", "swing rhythmically"

Torso:  
Describe posture or motion of the torso (e.g., bending, twisting).  
Avoid overused phrases unless accurate.  
✅ Examples: "bends at waist", "leans slightly left", "remains upright for balance"

Legs:  
Describe whether the legs move, support, or shift weight.  
✅ Examples: "remain planted", "alternate stepping", "bend slightly for balance"

Start:  
Initial body pose.  
✅ Examples: "stand upright with arms at sides", "feet together, relaxed posture"

Middle:  
Main/peak motion.  
✅ Examples: "raise one arm overhead", "twist torso while leaning forward"

End:  
Final pose.  
✅ Examples: "return to upright position", "arms lowered"

Global Desc:  
Short natural summary (≤12 words).  
✅ Examples: "raise one hand overhead while body remains still"

VR Scenario:  
Describe what this action would look like in a virtual environment.  
✅ Examples: "raising hand in a virtual classroom", "walking in a VR corridor"

Interaction:  
If the action is in this list:  
['walking', 'running', 'jumping', 'bending down', 'stand', 'squatting', 'raising hand']  
→ write exactly `"unrelated"`  
Otherwise, describe interaction with VR elements  
✅ Example: "use controller to pick up a box"

---

Now write a 10-line structured description for the action: "{action}"
"""







from action_data import actions, motion_type_dict, jumping_dict, hand_usage_dict, limbs_involved_dict



# 存储结果的列表
results = []

# 循环处理每个动作
for action_name, action_desc in actions:
    print(f"Generating description for: {action_name}")

    motion_type = motion_type_dict.get(action_name, 'unknown')
    jumping = jumping_dict.get(action_name, 'no')
    hand_usage = hand_usage_dict.get(action_name, 'unknown')
    limbs_involved = limbs_involved_dict.get(action_name, 'unknown')

    filled_prompt = prompt_template2.format(
        action=action_name,
        action_list=actions
    )

    try:
        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': filled_prompt}
            ]
        )

    except Exception as e:
        print(f"Failed to generate for {action_name}: {e}")

    content = response.choices[0].message.content
    print(content)

    parts = content.split('\n')
    desc_dict = {'Action': action_name}  # ✅ 保存为动作名
    for part in parts:
        if part.startswith('Head'):
            desc_dict['Head'] = part[6:].strip()
        elif part.startswith('Hands'):
            desc_dict['Hands'] = part[7:].strip()
        elif part.startswith('Torso'):
            desc_dict['Torso'] = part[7:].strip()
        elif part.startswith('Legs'):
            desc_dict['Legs'] = part[6:].strip()
        elif part.startswith('Start'):
            desc_dict['Start'] = part[7:].strip()
        elif part.startswith('Middle'):
            desc_dict['Middle'] = part[8:].strip()
        elif part.startswith('End'):
            desc_dict['End'] = part[5:].strip()
        elif part.startswith('Global Desc'):
            desc_dict['Global Desc'] = part[13:].strip()
        elif part.startswith('VR Scenario'):
            # desc_dict['VR Scenario'] = part[13:].strip()
            desc_dict['VR Scenario'] = action_desc
        elif part.startswith('Interaction'):
            desc_dict['Interaction'] = part[13:].strip()
    
    results.append(desc_dict)
    print("Done.")

    time.sleep(2)

        
# 转换为 DataFrame 并保存到 Excel
df_columns = ['Action', 'Head', 'Hands', 'Torso', 'Legs', 'Start', 'Middle', 'End', 'Global Desc', 'VR Scenario', 'Interaction']
df = pd.DataFrame(results, columns=df_columns)

output_file = '/home/yanghao/SummerRA/Code/ZSAR/My/supplements/hku_test.xlsx'
df.to_excel(output_file, index=False)

print(f"Descriptions saved to {output_file}")