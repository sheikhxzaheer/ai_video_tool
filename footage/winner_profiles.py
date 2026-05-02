import json
import os
import re
import time
import logging
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
        
    genai.configure(api_key=api_key) # type: ignore
    model = genai.GenerativeModel("gemini-2.5-flash") # type: ignore
    
    logging.info(f"Gemini API connecting for: {Path(video_path).name}...")
    try:
        video_file = genai.upload_file(video_path) # type: ignore
        
        logging.info("Waiting for Google servers to process the video...")
        while video_file.state.name == "PROCESSING":
            time.sleep(3)
            video_file = genai.get_file(video_file.name) # type: ignore
            
        if video_file.state.name == "FAILED":
            logging.error(f"Google failed to process video: {Path(video_path).name}")
            return None
            
        response = model.generate_content([video_file, WINNER_ANALYSIS_PROMPT])
        
        raw_text = re.sub(r"```json\s*|```\s*", "", (response.text or "").strip())
        features = json.loads(raw_text)
        
        logging.info(f"Success! DNA extracted for {Path(video_path).name}")
        return features
        
    except Exception as e:
        logging.error(f"Gemini API Error for {Path(video_path).name}: {e}")
        return None

def update_brain_with_dna(features, filename):
    if not features:
        return
        
    brain = load_brain()
    
    if "processed_videos" not in brain:
        brain["processed_videos"] = [] # type: ignore
        
    if filename not in brain["processed_videos"]:
        brain["processed_videos"].append(filename) # type: ignore

    all_good_tags = []
    for key in ["structural_tags", "style_keywords", "visual_keywords", "mood"]:
        tags = features.get(key, [])
        all_good_tags.extend(tags)
    
    WINNER_BOOST = 0.15
    for tag in all_good_tags:
        if tag:
            clean_tag = tag.strip().lower()
            current_score = brain.get("winner_tags", {}).get(clean_tag, 0.0)
            brain["winner_tags"][clean_tag] = round(current_score + WINNER_BOOST, 4)
            
    with open("learning_weights.json", "w") as f:
        json.dump(brain, f, indent=2)
    
    logging.info(f"Brain updated successfully with DNA from {filename}")

def scan_all_winners(winners_folder="Winners"):
    if not os.path.exists(winners_folder):
        logging.warning(f"Folder '{winners_folder}' not found. Aborting scan.")
        return

    video_files = [f for f in os.listdir(winners_folder) if f.endswith(".mp4")]
    if not video_files:
        logging.warning(f"Folder '{winners_folder}' is empty. No videos to scan.")
        return

    logging.info(f"Starting batch scan. Found {len(video_files)} Winner videos.")
    
    brain = load_brain()
    processed_list = brain.get("processed_videos", [])

    for filename in video_files:
        if filename in processed_list:
            logging.info(f"Skipping: '{filename}' (Already in processed memory)")
            continue
            
        filepath = os.path.join(winners_folder, filename)
        dna_features = analyze_winner_video(filepath)
        
        if dna_features:
            update_brain_with_dna(dna_features, filename)
            logging.info("Applying 15s rate-limit cooldown to protect API quotas...")
            time.sleep(15)

    print("\n" + "=" * 50)
    print("Winner DNA saved to 'learning_weights.json'")
    print("\nTop 10 Strongest DNA Tags (AI's Favorite):")
    
    updated_brain = load_brain()
    sorted_tags = sorted(updated_brain.get("winner_tags", {}).items(), key=lambda x: -x[1])
    
    for tag, score in sorted_tags[:10]:
        bar = "█" * int(score * 20)
        print(f"  {tag:<25} {score:.2f}  {bar}")
    print("=" * 50)
    
    logging.info("Batch scanning complete.")

if __name__ == "__main__":
    scan_all_winners("Winners")