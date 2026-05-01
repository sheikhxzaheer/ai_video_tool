import json
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("ai_learning_loop.log"),
        logging.StreamHandler()
    ]
)

def load_brain(filepath="learning_weights.json"):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON: {e}")
    else:
        logging.warning(f"Can't find Brain file '{filepath}'. Starting with new empty brain.")

    return {"winner_tags": {}, "qa_rejected_tags": {}}

def save_rejection(bad_video_data, filepath="learning_weights.json"):
    brain = load_brain(filepath)
    bad_tags = bad_video_data.get("structural_tags", []) + bad_video_data.get("visual_keywords", [])
    REJECT_PENALTY = -0.15
    
    for tag in bad_tags:
        if tag:
            current_score = brain["qa_rejected_tags"].get(tag, 0.0)
            brain["qa_rejected_tags"][tag] = round(current_score + REJECT_PENALTY, 4)
            
    with open(filepath, "w") as f:
        json.dump(brain, f, indent=2)
    logging.info(f"🛑 REJECTED: Video rejected. Applied {REJECT_PENALTY} penalty to tags: {bad_tags}")

def save_winner(good_video_data, filepath="learning_weights.json"):
    brain = load_brain(filepath)
    good_tags = good_video_data.get("structural_tags", []) + good_video_data.get("visual_keywords", [])
    WINNER_BOOST = 0.15
    
    for tag in good_tags:
        if tag:
            current_score = brain["winner_tags"].get(tag, 0.0)
            brain["winner_tags"][tag] = round(current_score + WINNER_BOOST, 4)
            
    with open(filepath, "w") as f:
        json.dump(brain, f, indent=2)
    logging.info(f"✅ SAVED WINNER: Video accepted. Applied {WINNER_BOOST} boost to tags: {good_tags}")