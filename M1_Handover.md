# 🎬 AI Video Editing Tool — Milestone 1 Handover
**Prepared for:** Roman Lutsenko  
**Prepared by:** Sheikh Zaheer  
**Date:** May 2026  

---

## 📦 Project Files

| Source | Link | Contains |
|---|---|---|
| **Google Drive** | https://drive.google.com/drive/folders/1QyfvQE3DV0_p097J1FkM7o9u5l6wGa76?usp=sharing | Full project with Brands folder, .env, database, all assets |
| **GitHub** | https://github.com/sheikhxzaheer/ai_video_tool.git | Source code only (no large files) |

> **Recommended:** Download from Google Drive for the complete setup. GitHub is for code reference only.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

---

## 🧠 1. The AI Brain — Learning & Extraction System

### Elite DNA Extraction
The AI now analyzes winner videos like a Creative Director, not a keyword scraper. Each winning video is sent to Gemini with a carefully engineered prompt that extracts **5 precise creative traits** from the footage:

- `structural_tags` — the narrative role of the clip (e.g. Problem Hook, Product Intro, CTA)
- `visual_keywords` — specific objects and actions visible on screen
- `style_keywords` — cinematography style (e.g. Handheld, Fast-paced, Cinematic)
- `environment` — setting context (e.g. indoor, bedroom, studio)
- `mood` — emotional tone (e.g. energetic, calm, aspirational)

This gives the system a rich, multi-dimensional understanding of what makes your winning content work.

---

### Asymmetric Learning Weights
The system learns at different speeds intentionally:

- **Winner Boost:** `+0.15` — slow and measured, preserving variety in future selections
- **Reject Penalty:** `−0.25` — fast and decisive, immediately reducing bad clip patterns

This asymmetry means the system is cautious about locking in favorites but quick to eliminate what you dislike.

---

### Smart Normalization — The Intelligence Lock
All memory weights are hard-capped:
- Winner tags: maximum `+1.0`
- Rejection tags: minimum `−1.0`

Without this cap, an old strong preference could overpower the semantic search engine entirely. The lock ensures the AI always balances **what it has learned** with **what the current script is actually asking for**.

---

### Time-Decay Memory
Every 7 days, the system automatically applies a `0.90` decay factor to all rejection penalties. This means:

- A one-time mistake fades over time
- Repeated rejections stay strong
- The AI adapts to new trends without being held back by old decisions
- **Winner DNA from trained videos is permanent and never decays**

---

### Automated Winner Scanning
One button in the UI triggers a full recursive scan of all `Winners/` folders across every brand. The system:
- Finds all unprocessed winner videos automatically
- Sends each to Gemini for DNA extraction
- Applies rate-limiting cooldowns to protect API quotas
- Tracks processed videos so nothing is analyzed twice

---

## ⚙️ 2. The Core Matching Engine — Backend Logic

### Hybrid Reranking Engine
Matching happens in two stages:

**Stage 1 — Semantic Search (ChromaDB):** Finds the top 25 candidate clips based on text meaning of the voiceover segment.

**Stage 2 — AI Brain Reranking:** The custom `reranker.py` applies your learned DNA weights on top of the semantic results. Final score formula:

```
Final Score = Semantic Score + Winner Tag Boost + QA Penalty
```

Every clip displays its full score breakdown:
```
clip.mp4 | Score: 0.83 (Semantic: 0.71, Winner: +0.27, QA: -0.15)
```

---

### Strict Clipping Protocol — The Sync Saver
A critical safety layer ensures audio and video never fall out of sync. If the AI selects a 49-second clip but the voiceover segment is only 2 seconds long, the backend automatically trims the clip to exactly 2 seconds. **Sync is guaranteed at all times.**

---

### Deep Alternative Generation
For every segment, the system pre-calculates and stores **3 backup clips** in memory alongside the primary selection. These are instantly available when you click Reject — no reprocessing, no waiting.

---

## 🖥️ 3. The Enterprise UI — Frontend Experience

### Split-Screen Segment Inspector
The output panel is structured like a professional editing interface. Each segment shows:
- Left side: voiceover text, AI score explanation, and action controls
- Right side: visual clip data and the inline video player

---

### Inline Source Monitor with Auto-Jump Playback
Instead of showing a raw file path, the UI embeds an actual video player inside the interface. When you click play, the video **automatically jumps to the exact cut-in point** the AI selected — not from the beginning of the file. You see precisely the frame the AI chose.

---

### Visual Timeline Progress Bar
Raw timecode text (`In: 2.3s | Out: 5.1s`) has been replaced with a Premiere-style visual bar. A colored block shows exactly which portion of the source clip is being used, scaled to the total clip duration.

---

### One-Click Reject & Swap
When you click ❌ Reject on any segment:
1. The rejected clip's feature tags receive a `−0.25` penalty instantly
2. The updated weights are saved to `learning_weights.json`
3. The next best alternative clip appears on screen immediately
4. No audio reprocessing required

The system learns from the **actual features** of the rejected clip — not just a generic flag. Every rejection makes future selections smarter.

---

### Alternative Availability Counter
The UI displays how many backup clips are ready for each segment (e.g. `🔄 3 Backup Clips Ready`). You always know whether you have options before deciding to reject.

---

### Smart Duration Warning
If the AI selects a clip that is shorter than the voiceover segment it needs to cover, the UI displays a yellow warning badge:

> ⚠️ Clip is shorter than voiceover. Extra B-roll may be required.

This prevents timeline gaps from going unnoticed.

---

### Live Brain Dashboard
Visible at all times in the output panel, the Brain Status section shows:
- **Top 3 Favourite Styles** — what the AI currently prefers most
- **Top 3 Avoided Styles** — what the AI is currently penalizing

This makes the AI's decision-making fully transparent and inspectable at a glance.

---

## 🗂️ 4. Folder Structure for Future Brands

To add a new brand niche, simply follow this structure:

```
Brands/
  Pillow/
    Winners/          ← place final edited winning videos here
  Energy/
    Winners/
  Mold/
    Winners/
  HearingAids/        ← new brand
    Winners/          ← add winning videos when ready
```

Once videos are added, click **"Train on New Winners"** in the sidebar. The system handles everything automatically.

> **Note:** 3 Facial Mask videos (`Mask-99 SS1.mov`, `Mask-162 SS1.mp4`, `Mask-89(2)SS1.mp4`) could not be processed. These files fail to load even directly on the Gemini website, so the issue is with the video files themselves. All other winners trained successfully. These can be re-exported and trained later.

---

## 🎬 5. Export Engine — Delivery Pipeline

### Dynamic FCP XML Generator
The JSON output is converted into a Premiere Pro-compatible FCP XML file using a custom export engine. The timeline is assembled automatically with all clips placed and synced.

### Variable Framerate Support
An FPS selector in the UI allows you to choose 24, 25, 30, or 60 FPS before exporting. The engine converts all timecodes to exact frame counts automatically. Default is **30 FPS** as discussed.

### Multi-Track Timeline Mapping
The exported XML places:
- **Matched video clips** on Video Track (V1)
- **Voiceover audio** on Audio Track (A1)
- **Subtitle text layers** as editable Essential Graphics on the text track

Everything is pre-synced. Open the XML in Premiere and the edit is already assembled.

---

## 🧹 6. Engineering Standards

### Clean Dependencies
`requirements.txt` contains only the libraries that are actually used — all pinned to stable versions. No bloat, no experimental packages. The same enterprise-grade tools from the original foundation.

### Automatic Backup System
Every time `learning_weights.json` is modified, a `.bak` backup is created automatically. If the file is ever corrupted, the system restores from the backup without data loss.

### Structured Logging
All system activity — winner training, rejections, time decay cycles — is written to `ai_learning_loop.log`. A full audit trail of everything the AI has learned and when.

---

## ✅ M1 Delivery Summary

| Feature | Status |
|---|---|
| 3-signal weighted scoring engine | ✅ Complete |
| Winner DNA extraction via Gemini | ✅ Complete — 20 videos trained |
| QA reject button with real-time learning | ✅ Complete |
| Feature-level penalties (not file-level) | ✅ Complete |
| Score explanation per clip | ✅ Complete |
| Time decay on rejection weights | ✅ Complete |
| Score normalization (caps at ±1.0) | ✅ Complete |
| 3 pre-loaded alternatives per segment | ✅ Complete |
| Inline video player with auto-jump | ✅ Complete |
| Visual timeline progress bar | ✅ Complete |
| Duration warning badge | ✅ Complete |
| Brain status dashboard | ✅ Complete |
| Automatic backup system | ✅ Complete |
| Structured activity logging | ✅ Complete |
| FCP XML export engine (M2) | ✅ Complete |
