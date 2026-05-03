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
You are an elite Direct-Response Video Editor and Creative Strategist. Your task is to perform a deep "DNA Extraction" on this winning video ad. 
We are feeding this data into an advanced AI scoring system, so your tags must be highly precise, descriptive, and analytical.

Analyze the video's core mechanics—why it captures attention, its pacing, its visual hooks, and its emotional resonance. 

Guidelines for Tagging:
- Use clear, descriptive terms connected by underscores (e.g., "fast_paced_montage", "intimate_ugc").
- Avoid vague or generic words like "good", "happy", "nice", "video".
- "structural_tags" should define the marketing framework (e.g., "Problem_Hook", "Social_Proof", "Product_Demo", "Us_vs_Them").
- "visual_keywords" should be literal actions or objects (e.g., "pointing_at_camera", "applying_cream", "green_screen", "holding_phone").
- "style_keywords" should define the camera work and editing (e.g., "jump_cuts", "handheld_shaky", "text_heavy", "split_screen").
- Keep all tags concise (1 to 3 words maximum).

Return ONLY valid JSON — no markdown formatting, no explanations, no code blocks:
{
  "structural_tags": ["e.g. Problem_Hook, Face_to_Camera, Product_Intro"],
  "visual_keywords": ["e.g. holding_phone, drinking, running"],
  "style_keywords": ["e.g. Handheld, Fast_paced, Low_res_UGC, Cinematic"],
  "environment": ["e.g. indoor, bedroom, street"],
  "mood": ["e.g. energetic, calm, chaotic, urgent, trustworthy"]
}
"""

def find_all_winners_folders(root_dir="Brands"):
    root_path = Path(root_dir)
    if not root_path.exists():
        return []
    winners_folders = [str(p) for p in root_path.rglob("**/Winners") if p.is_dir()]
    
    return winners_folders
    
def analyze_winner_video(video_path: str, ui_callback=None):
    def send_status_update(msg):
        logging.info(msg)
        if ui_callback:
            ui_callback(msg)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("Can't find GEMINI_API_KEY in .env file!")
        return None
        
    genai.configure(api_key=api_key) # type: ignore
    model = genai.GenerativeModel("gemini-2.5-flash") # type: ignore
    
    send_status_update(f"Uploading to Gemini: {Path(video_path).name}...")
    try:
        video_file = genai.upload_file(video_path) # type: ignore
        
        send_status_update("Waiting for Gemini to analyze video...")
        while video_file.state.name == "PROCESSING":
            time.sleep(3)
            video_file = genai.get_file(video_file.name) # type: ignore
            
        if video_file.state.name == "FAILED":
            send_status_update(f"Gemini failed to process: {Path(video_path).name}")
            return None
            
        response = model.generate_content([video_file, WINNER_ANALYSIS_PROMPT])
        
        raw_text = re.sub(r"```json\s*|```\s*", "", (response.text or "").strip())
        features = json.loads(raw_text)
        
        send_status_update(f"DNA extracted: {Path(video_path).name}")
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

def scan_all_winners(winners_folder="Winners", ui_callback=None):
    def send_status_update(msg):
        logging.info(msg)
        if ui_callback:
            ui_callback(msg)
    if not os.path.exists(winners_folder):
        logging.warning(f"Folder '{winners_folder}' not found. Aborting scan.")
        return

    folder_path = Path(winners_folder)
    video_files = list(folder_path.rglob("*.mp4"))

    if not video_files:
        logging.warning(f"Folder '{winners_folder}' is empty or has no .mp4 files. Skipping.")
        return

    send_status_update(f"Found {len(video_files)} winner video(s) in {winners_folder}")

    brain = load_brain()
    processed_list = brain.get("processed_videos", []) if brain else []
    
    for filepath_obj in video_files:
        filepath = str(filepath_obj)
        filename = filepath_obj.name

        if filename in processed_list:
            send_status_update(f"Skipping '{filename}' (already processed)")
            continue

        dna_features = analyze_winner_video(filepath, ui_callback)
        if dna_features:
            update_brain_with_dna(dna_features, filename)
            send_status_update("Cooldown: 15s rate-limit to protect API")
            time.sleep(15)
            send_status_update("Resuming analysis...")
    print("\n" + "=" * 50)
    print(f"Winner DNA from {winners_folder} saved to 'learning_weights.json'")
    print("\nTop 10 Strongest DNA Tags (AI's Favorite):")
    
    updated_brain = load_brain()
    sorted_tags = sorted(updated_brain.get("winner_tags", {}).items(), key=lambda x: -x[1])
    
    for tag, score in sorted_tags[:10]:
        bar = "█" * int(score * 20)
        print(f"  {tag:<25} {score:.2f}  {bar}")
    print("=" * 50)
    
    send_status_update(f"Batch scan complete for {winners_folder}")