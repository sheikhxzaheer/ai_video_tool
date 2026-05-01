import json
import os
import re
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Yahan hum apne brain ko import kar rahe hain (Fixing NameError)
from footage.qa_store import load_brain

load_dotenv()

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
    api_key = "AIzaSyDTlvWzBbZMH7zsUxqqerJSi2nvu0sgcb8"
    if not api_key:
        print("Error: GEMINI_API_KEY nahi mili .env file mein!")
        return None
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    print(f"\n🎥 Gemini is analyzing: {Path(video_path).name}...")
    try:
        video_file = genai.upload_file(video_path)
        
        print("Waiting for Gemini to process the video...")
        while video_file.state.name == "PROCESSING":
            time.sleep(3)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == "FAILED":
            print("Video process nahi hui.")
            return None
            
        response = model.generate_content([video_file, WINNER_ANALYSIS_PROMPT])
        
        raw_text = re.sub(r"```json\s*|```\s*", "", (response.text or "").strip())
        features = json.loads(raw_text)
        
        print("✅ Success! DNA extracted.")
        return features
        
    except Exception as e:
        print(f"❌ Error aagaya bhai: {e}")
        return None

def update_brain_with_dna(features, filename):
    """Yeh naya function DNA save karega aur yaad rakhega ki kaunsi video ho chuki hai"""
    if not features:
        return
        
    brain = load_brain()
    
    # Agar JSON mein 'processed_videos' ki list nahi hai, toh banao
    if "processed_videos" not in brain:
        brain["processed_videos"] = []
        
    # Video ka naam processed list mein daal do
    if filename not in brain["processed_videos"]:
        brain["processed_videos"].append(filename)

    all_good_tags = []
    for key in ["structural_tags", "style_keywords", "visual_keywords", "mood"]:
        tags = features.get(key, [])
        all_good_tags.extend(tags)
    
    WINNER_BOOST = 0.15
    for tag in all_good_tags:
        if tag:
            current_score = brain.get("winner_tags", {}).get(tag, 0.0)
            brain["winner_tags"][tag] = round(current_score + WINNER_BOOST, 4)
            
    with open("learning_weights.json", "w") as f:
        json.dump(brain, f, indent=2)

def scan_all_winners(winners_folder="Winners"):
    if not os.path.exists(winners_folder):
        print(f"Bhai, '{winners_folder}' naam ka folder nahi mila!")
        return

    video_files = [f for f in os.listdir(winners_folder) if f.endswith(".mp4")]
    if not video_files:
        print(f"Folder '{winners_folder}' mein koi video nahi hai.")
        return

    print(f"Total {len(video_files)} Winner videos mili hain. Scanning start kar rahe hain...\n")
    
    # Brain load karo taaki check kar sakein kaunsi videos pehle se ho chuki hain
    brain = load_brain()
    processed_list = brain.get("processed_videos", [])

    for filename in video_files:
        # THE FIX: Agar video pehle ho chuki hai, toh usko SKIP karo!
        if filename in processed_list:
            print(f"⏩ Skipping: '{filename}' (Yeh pehle se analyze ho chuki hai)")
            continue
            
        filepath = os.path.join(winners_folder, filename)
        dna_features = analyze_winner_video(filepath)
        
        if dna_features:
            update_brain_with_dna(dna_features, filename)
            
            # THE FIX: API Limit (429 Error) bachane ke liye 15 seconds ruko
            print("⏳ API limit bachane ke liye 15 seconds wait kar rahe hain...")
            time.sleep(15)

    print("\n" + "=" * 50)
    print("Winner DNA 'learning_weights.json' mein save ho gaya hai!")
    print("\nTop 10 Strongest DNA Tags (AI's Favorite):")
    
    updated_brain = load_brain()
    sorted_tags = sorted(updated_brain.get("winner_tags", {}).items(), key=lambda x: -x[1])
    
    for tag, score in sorted_tags[:10]:
        bar = "█" * int(score * 20)
        print(f"  {tag:<25} {score:.2f}  {bar}")
    print("=" * 50)


if __name__ == "__main__":
    # Tumhare actual folder ka path
    scan_all_winners("Brands/Pillow/winners")