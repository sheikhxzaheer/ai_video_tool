import google.generativeai as genai
import time
import os
from dotenv import load_dotenv
import json
from llm.prompt import PREDEFINED_TAGS

load_dotenv()
# 1. CONFIGURE FIRST
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

SYSTEM_PROMPT = """
    Act as a professional Video Editor. Your task is to extract the "Semantic Core" of this video.
    
    The "Semantic Core" is the segment where the actual value, action, or expertise is demonstrated.
    
    ### DETAILED HERO MOMENT CRITERIA:
    Identify and KEEP the following "Hero Moments":
    1. THE PROCESS/LABOR: Any professional action (e.g., a scientist in a lab mixing liquids, a developer typing code, a craftsman carving wood, a chef chopping). The act of "doing the work" IS the hero moment.
    2. THE REVEAL/RESULT: The moment a transformation is shown (e.g., a dirty surface becoming clean, a "Before/After" side-by-side, a final product shot).
    3. PRODUCT INTERACTION: Using the product (e.g., clicking a button, opening a lid, applying cream, pouring a drink).
    4. EMOTIONAL CLIMAX: A clear facial expression (e.g., a "Wow" reaction, a look of frustration during a "Problem State," or a smile of satisfaction).
    5. KEY INFORMATION: A "Talking Head" segment where the speaker makes a main point or a "Call to Action." (NOTE: Subject to the "Visual Priority Rule" below).
    
    ### TRIMMING & CLEANING RULES:
    1. THE VISUAL PRIORITY RULE: In segments where a subject is speaking, prioritize and keep only shots that capture the product itself (e.g., how it is held, handled, or used). Remove segments consisting only of spoken dialogue or "talking head" footage without active product interaction.
    
    2. CUT "Setup & Calibration": Remove frames where the subject is adjusting their hair, looking for the camera, waiting for a "Go" signal, or camera focus-hunting.

    3. CUT "Post-Action Dead Air": Remove frames after the main action is finished (e.g., subject looking away, walking out of frame, or the hand lingering on the product for too long).

    4. THE "NO-MISTAKE" RULE: If the video starts exactly with the action and ends exactly when the action stops, DO NOT trim anything. Keep 100% of the duration.

    HERE IS THE 38-TAG LIST:
    {PREDEFINED_TAGS}

    
    ### OUTPUT FORMAT (Strict JSON):
    {{
    "detailed_description": "What is the core meaning of this video? (5-7 sentences MAX)",
    "segments": [
        {{
        "start_time": "MM:SS.MS",
        "end_time": "MM:SS.MS",
        "structural_tags": ["Select from your 38-tag list"],
        "visual_keywords": ["Specific objects/actions"],
        "why_kept": "Explain why this segment is the Semantic Core."
        }}
    ]
    }}
"""

SYSTEM_PROMPT = SYSTEM_PROMPT.format(PREDEFINED_TAGS=PREDEFINED_TAGS.values())

# 2. Use raw strings for Windows paths to avoid escape character issues
video_path = r"746f53-billo-169883-orig.mp4"

print("Uploading file...")
video_file = genai.upload_file(path=video_path)

# 3. Wait for processing
while video_file.state.name == "PROCESSING":
    print(".", end="", flush=True)
    time.sleep(2)
    video_file = genai.get_file(video_file.name)

# 4. Generate content
model = genai.GenerativeModel(model_name="gemini-2.5-pro")
response = model.generate_content([
    video_file,
    SYSTEM_PROMPT
])

response_json = response.text.split("```json")[1].split("```")[0]
response_json = json.loads(response_json)
with open("response_1.json", "w") as f:
    json.dump(response_json, f, indent=4)

print("\n" + response.text)