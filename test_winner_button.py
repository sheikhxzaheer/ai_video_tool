from footage.reranker import save_winner

hit_video_from_ui = {
    "path": "C:\\videos\\cozy_bedroom_morning.mp4",
    "structural_tags": ["morning", "cozy"],
    "visual_keywords": ["sunlight", "fluffy_pillow", "smiling"]
}

print("Roman ne 'Mark as Winner' button dabaya...")

save_winner(hit_video_from_ui)

print("Check karo apni learning_weights.json file!")