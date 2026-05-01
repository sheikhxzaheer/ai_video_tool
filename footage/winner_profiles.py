import json
import logging
import os
import re
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

from footage.qa_store import load_brain

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("ai_learning_loop.log"),
        logging.StreamHandler()
    ]
)

WINNER_ANALYSIS_PROMPT = """
You are analyzing a winning video ad. Extract its visual features for a scoring system.
Focus on the style, pacing, and vibe (the DNA).
Return ONLY valid JSON — no markdown, no explanation:
{
  "structural_tags": ["e.g. Problem_Hook, Face_to_Camera, Product_Intro"],
  "visual_keywords": ["e.g. holding_phone, drinking, running"],
  "style_keywords": ["e.g. Handheld, Fast_paced, Low_res_UGC, Cinematic"],
  "environment": ["e.g. indoor, bedroom, street"],
  "mood": ["e.g. energetic, calm, chaotic"]
}
"""

def analyze_winner_video(video_path: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("Can't find GEMINI_API_KEY in .env file!")
        return None
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    logging.info(f"Gemini API connecting for: {Path(video_path).name}...")
    try:
        video_file = genai.upload_file(video_path)
        logging.info("Waiting for Google servers to process the video...")
            
        if video_file.state.name == "FAILED":
            logging.error(f"Google failed to process video: {Path(video_path).name}")
            return None
            
        response = model.generate_content([video_file, WINNER_ANALYSIS_PROMPT])
        
        raw_text = re.sub(r"```json\s*|```\s*", "", (response.text or "").strip())
        features = json.loads(raw_text)
        
        logging.info(f"Success! DNA extracted for {Path(video_path).name}")
        return features
        
    except Exception as e:
        print(f"❌ Error aagaya bhai: {e}")
        return None

def update_brain_with_dna(features, filename):
    """Yeh naya function DNA save karega aur yaad rakhega ki kaunsi video ho chuki hai"""
    if not features:
        return
        
    brain = load_brain()

def scan_all_winners(winners_folder="Winners"):
    if not os.path.exists(winners_folder):
        print(f"Bhai, '{winners_folder}' naam ka folder nahi mila!")
        return

    video_files = [f for f in os.listdir(winners_folder) if f.endswith(".mp4")]
    if not video_files:
        print(f"Folder '{winners_folder}' mein koi video nahi hai.")
        return

    


if __name__ == "__main__":
    scan_all_winners("Brands/Pillow/winners")