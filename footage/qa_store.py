import json
import os
import logging
import shutil
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("ai_learning_loop.log"),
        logging.StreamHandler()
    ]
)

DECAY_INTERVAL_DAYS = 7
DECAY_FACTOR = 0.90
MIN_WEIGHT = 0.01

def _apply_time_decay(brain, filepath):
    today_str = datetime.now(timezone.utc).date().isoformat()
    last_decay = brain.get("last_decay_date", None)
    if last_decay:
        last_date = datetime.fromisoformat(last_decay).date()
        today_date = datetime.now(timezone.utc).date()
        days_since = (today_date - last_date).days
        if days_since < DECAY_INTERVAL_DAYS:
            return brain

    for tag in list(brain.get("winner_tags", {}).keys()):
        new_val = round(brain["winner_tags"][tag] * DECAY_FACTOR, 4)
        if abs(new_val) < MIN_WEIGHT:
            del brain["winner_tags"][tag]
        else:
            brain["winner_tags"][tag] = new_val

    for tag in list(brain.get("qa_rejected_tags", {}).keys()):
        new_val = round(brain["qa_rejected_tags"][tag] * DECAY_FACTOR, 4)
        if abs(new_val) < MIN_WEIGHT:
            del brain["qa_rejected_tags"][tag]
        else:
            brain["qa_rejected_tags"][tag] = new_val

    brain["last_decay_date"] = today_str

    if os.path.exists(filepath):
        shutil.copy(filepath, filepath + ".bak")
    with open(filepath, "w") as f:
        json.dump(brain, f, indent=2)

    logging.info(f"TIME DECAY applied. Factor: {DECAY_FACTOR}. Next decay in {DECAY_INTERVAL_DAYS} days.")

    return brain


def load_brain_with_decay(filepath="learning_weights.json"):
    brain = load_brain(filepath)
    brain = _apply_time_decay(brain, filepath)
    return brain



def load_brain(filepath="learning_weights.json"):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading JSON: {e}")
            backup_path = filepath + ".bak"
            if os.path.exists(backup_path):
                logging.info(f"Restoring from backup file: {backup_path}")
                try:
                    with open(backup_path, "r") as f:
                        return json.load(f)
                except Exception as backup_error:
                    logging.error(f"Backup file is also corrupted: {backup_error}")
    else:
        logging.warning(f"Can't find Brain file '{filepath}'. Starting with new empty brain.")

    return {"winner_tags": {}, "qa_rejected_tags": {}, "processed_videos": []}

def save_rejection(bad_video_data, filepath="learning_weights.json"):
    brain = load_brain(filepath)
    bad_tags = bad_video_data.get("structural_tags", []) + bad_video_data.get("visual_keywords", [])
    REJECT_PENALTY = -0.15
    
    for tag in bad_tags:
        if tag:
            clean_tag = tag.strip().lower()
            current_score = brain.get("qa_rejected_tags", {}).get(clean_tag, 0.0)
            brain["qa_rejected_tags"][clean_tag] = round(current_score + REJECT_PENALTY, 4)
    if os.path.exists(filepath):
        shutil.copy(filepath, filepath + ".bak")
            
    with open(filepath, "w") as f:
        json.dump(brain, f, indent=2)
    logging.info(f"🛑 REJECTED: Video rejected. Applied {REJECT_PENALTY} penalty.")

def save_winner(good_video_data, filepath="learning_weights.json"):
    brain = load_brain(filepath)
    good_tags = good_video_data.get("structural_tags", []) + good_video_data.get("visual_keywords", [])
    WINNER_BOOST = 0.15
    
    for tag in good_tags:
        if tag:
            clean_tag = tag.strip().lower()
            current_score = brain.get("winner_tags", {}).get(clean_tag, 0.0)
            brain["winner_tags"][clean_tag] = round(current_score + WINNER_BOOST, 4)
    if os.path.exists(filepath):
        shutil.copy(filepath, filepath + ".bak")
            
    with open(filepath, "w") as f:
        json.dump(brain, f, indent=2)
    logging.info(f"✅ SAVED WINNER: Video accepted. Applied {WINNER_BOOST} boost.")

def clean_existing_brain(filepath="learning_weights.json"):
    brain = load_brain(filepath)
    cleaned_brain = {
        "winner_tags": {},
        "qa_rejected_tags": {},
        "processed_videos": brain.get("processed_videos", [])
    }
    for tag, score in brain.get("winner_tags", {}).items():
        clean_tag = tag.strip().lower()
        current_score = cleaned_brain["winner_tags"].get(clean_tag, 0.0)
        cleaned_brain["winner_tags"][clean_tag] = round(current_score + score, 4)
    for tag, score in brain.get("qa_rejected_tags", {}).items():
        clean_tag = tag.strip().lower()
        current_score = cleaned_brain["qa_rejected_tags"].get(clean_tag, 0.0)
        cleaned_brain["qa_rejected_tags"][clean_tag] = round(current_score + score, 4)
    if os.path.exists(filepath):
        shutil.copy(filepath, filepath + ".bak")
    with open(filepath, "w") as f:
        json.dump(cleaned_brain, f, indent=2)
    print("\n✅ Success! Brain cleaned and normalized. All tags have been standardized to lowercase and stripped of whitespace.")

if __name__ == "__main__":
    clean_existing_brain()