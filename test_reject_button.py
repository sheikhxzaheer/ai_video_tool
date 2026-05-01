from footage.reranker import save_rejection

video_from_ui = {
    "path": "C:\\videos\\beach_walking.mp4",
    "structural_tags": ["beach", "outdoor"],
    "visual_keywords": ["sand", "walking", "sun"]
}

print("Roman ne Reject button dabaya...")

# Hamare naye function ko call karo!
save_rejection(video_from_ui)

print("Check karo apni learning_weights.json file!")