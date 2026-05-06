import streamlit as st
import os
import json
import asyncio
import time
from pathlib import Path
from typing import Optional
from footage.qa_store import load_brain, save_rejection
from footage.winner_profiles import scan_all_winners, find_all_winners_folders

from dotenv import load_dotenv
load_dotenv()

from alignment.align_audio import load_model, align_audio, align_audio_with_script
from llm.tag_generator import generate_tags_async
from llm.cut_generator import insert_markers, segment_sentences
from processing.segment_refiner import extract_blocks, map_chunks_to_segments
from footage.indexer import build_footage_index
from footage.matcher import match_segments_to_footage


st.set_page_config(
    page_title="AI Video Editing Tool",
    page_icon="🎬",
    layout="wide")

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processing' not in st.session_state:
    st.session_state.processing = False


def init_models():
    """Initialize and cache models for faster processing."""
    if 'asr_model' not in st.session_state:
        with st.spinner("Loading Whisper model..."):
            st.session_state.asr_model = load_model()


async def process_pipeline(audio_path: str, script: Optional[str] = None) -> dict:
    """
    Main processing pipeline.
    
    Args:
        audio_path: Path to the voiceover audio file
        script: Optional script text for reference
        
    Returns:
        Structured data with segments and tags
    """
    st.info("Step 1/5: Transcribing and aligning audio...")
    asr_model = st.session_state.asr_model
    script_alignment = None
    if script:
        alignment_result = align_audio_with_script(audio_path, script, asr_model)
        word_timestamps = alignment_result["word_timestamps"]
        script_alignment = alignment_result.get("script_alignment") or None
    else:
        word_timestamps = align_audio(audio_path, asr_model)
    
    st.info("Step 2/5: LLM sentence segmentation + ASR time mapping...")
    # Prefer user-provided script for cleaner sentence boundaries; otherwise use ASR transcript.
    base_text = (script or " ".join(w.get("word", "") for w in word_timestamps)).strip()
    marked_sentences = await segment_sentences(
        base_text,
        model="claude-haiku-4-5",
    )
    sentence_chunks = extract_blocks(marked_sentences)
    if not sentence_chunks:
        # Fallback: treat whole text as one chunk
        sentence_chunks = [base_text] if base_text else []
    
    segments = map_chunks_to_segments(
        sentence_chunks,
        word_timestamps=word_timestamps,
        script_alignment=script_alignment,
    )
    for seg in segments:
        seg["section"] = None
        seg["subtitle"] = [seg["text"]]
    
    st.info("Step 3/5: Generating semantic tags with LLM...")
    segments_with_tags = await generate_tags_async(
        segments,
        provider=os.getenv("LLM_PROVIDER", "gemini"),
    )
    
    st.info("Step 4/5: Matching footage (vector search)...")
    segments_matched = match_segments_to_footage(segments_with_tags)
    
    video_length = max(seg['end'] for seg in segments_matched) if segments_matched else 0
    
    result = {
        "video_length": round(video_length, 2),
        "segments": segments_matched,
    }
    
    if script_alignment:
        result["script_alignment"] = script_alignment
    
    st.info("Step 5/5: Export ready!")
    return result

def display_brain_status():
    """UI Panel to show what the AI has learned so far."""
    st.subheader("🧠 AI Brain Status")
    brain = load_brain()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Learned Styles (Favorites)**")
        winners = sorted(brain.get("winner_tags", {}).items(), key=lambda x: -x[1])
        if winners:
            for tag, score in winners[:3]:
                st.success(f"📈 {tag} (+{score:.2f})")
        else:
            st.info("No winners saved yet.")
            
    with col2:
        st.markdown("**Avoided Styles (Rejected)**")
        rejected = sorted(brain.get("qa_rejected_tags", {}).items(), key=lambda x: x[1])
        if rejected:
            for tag, score in rejected[:3]:
                st.error(f"📉 {tag} ({score:.2f})")
        else:
            st.info("No rejections yet.")



def display_segments_with_alternatives(data):
    st.markdown("""
        <style>
        .red-button {
            background-color: #ff4444 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            border-radius: 0.25rem !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            font-size: 0.95rem !important;
        }
        .red-button:hover {
            background-color: #cc0000 !important;
        }
        .timeline-container {
            width: 100%; height: 10px; background-color: #2b2b36; 
            border-radius: 5px; display: flex; overflow: hidden; margin-top: 5px; margin-bottom: 15px;
        }
        .timeline-cut {
            background-color: #00d26a; border-radius: 5px; height: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader("🎬 Segment Inspector")
    for i, seg in enumerate(data['segments']):
        title_text = seg.get('text', '')[:60] + "..." if len(seg.get('text', '')) > 60 else seg.get('text', '')
        with st.expander(f"Segment {i+1} | {title_text}", expanded=False):
            matched = seg.get('matched_footage', {})
            alts = seg.get('alternatives', [])
            if matched:
                clip_path = matched.get('path', '')

                clip_name = Path(clip_path).name
                in_point_seconds = float(matched.get('in_point', 0))
                out_point_seconds = float(matched.get('out_point', 0))
                vo_duration = seg.get('end', 0) - seg.get('start', 0)
                clip_duration = out_point_seconds - in_point_seconds

                col_info, col_video = st.columns([1.2, 1])

                with col_info:
                    st.markdown("### Script & Timing")
                    st.write(f"**Voiceover:** *\"{seg.get('text', '')}\"*")
                    st.caption(f"**VO Timing:** {seg.get('start', 0)}s - {seg.get('end', 0)}s (Length: {round(vo_duration, 1)}s)")
                    
                    st.divider()
                    st.markdown("### AI Analysis")
                    st.info(f"**Reasoning:** {matched.get('score_explanation', 'N/A')}")
                    
                    matched_tags = matched.get('structural_tags', [])[:3] + matched.get('style_keywords', [])[:2]
                    if matched_tags:
                        st.caption(f"**Key Matches:** {', '.join(matched_tags)}")
                        
                    st.divider()

                    alts_count = len(alts)
                    if alts_count > 0:
                        st.markdown(f"**🔄 {alts_count} Backup Clips Ready**")
                        reject_label = f"❌ Reject & Swap Clip"
                    else:
                        reject_label = "❌ Reject (No backups left)"

                    if alts and st.button(reject_label, key=f"reject_btn_{i}", use_container_width=True):
                        bad_clip_data = {
                            "structural_tags": (
                                matched.get("structural_tags", []) +
                                matched.get("style_keywords", []) +
                                matched.get("environment", []) +
                                matched.get("mood", [])
                            ),
                            "visual_keywords": matched.get("visual_keywords", [])
                        }
                        save_rejection(bad_clip_data)
                        new_matched = alts.pop(0)
                        seg['matched_footage'] = new_matched
                        alts.append(matched)
                        seg['alternatives'] = alts
                        st.session_state.processed_data = data
                        st.rerun()

                with col_video:
                    try:
                        st.video(clip_path, start_time=in_point_seconds)
                    except Exception:
                        st.warning(f"⚠️ Video preview not available.")
                    st.caption(f"**File:** {clip_name}")

                    visual_total = max(out_point_seconds * 1.2, 15.0) 
                    in_pct = (in_point_seconds / visual_total) * 100
                    cut_pct = (clip_duration / visual_total) * 100

                    timeline_html = f"""
                    <div style="font-size: 12px; color: #bbb; display: flex; justify-content: space-between; margin-top: 10px;">
                        <span>In: <b>{round(in_point_seconds, 1)}s</b></span>
                        <span>Out: <b>{round(out_point_seconds, 1)}s</b></span>
                    </div>
                    <div class="timeline-container">
                        <div style="width: {in_pct}%; background-color: transparent;"></div>
                        <div class="timeline-cut" style="width: {cut_pct}%;"></div>
                    </div>
                    """
                    st.markdown(timeline_html, unsafe_allow_html=True)
                    if clip_duration < vo_duration:
                        st.warning(f"⚠️ Clip is shorter than VO. Extra B-roll required.")
                    else:
                        st.warning("No footage matched for this segment.")


def main():
    """Main Streamlit application."""
    st.title("🎬 AI Video Editing Tool")
   
    init_models()

    with st.sidebar:
        st.header("AI Brain Training")
        st.caption("Automatically finds & learns from all 'Winners' folders in Brands/")
        
        if st.button("Train on New Winners", type="primary"):
            with st.spinner("Scanning for Winners folders..."):
                live_status_box = st.empty()
                def update_live_status(msg):
                    live_status_box.info(f"⚡ **Live Activity:** {msg}")
                with st.spinner("Scanning for Winners folders..."):
                    winners_folders = find_all_winners_folders("Brands")
                    if not winners_folders:
                        st.warning(" No 'Winners' folders found in Brands/")
                    else:
                        st.info(f"Found {len(winners_folders)} Winners folder(s):")
                        for wf in winners_folders:
                            st.caption(f"  📂 {wf}")
                        try:
                            for winners_path in winners_folders:
                                update_live_status(f"Opening folder: {winners_path}")
                                scan_all_winners(winners_path, ui_callback=update_live_status)
                            live_status_box.success("✅ Training complete! AI Brain updated with all winners.")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
                            st.exception(e)
        st.divider()

        st.header("📁 Footage Index (Milestone-2)")
        st.caption("Build the segment-level vector index from your footage folder before processing.")
        footage_root = st.text_input(
            "Footage path",
            value="final-database",
            help="Path to footage folder (videos analyzed with Gemini)",
        )
        if st.button("🔨 Build Footage Index"):
            with st.spinner("Indexing footage (Gemini video analysis, embeddings)..."):
                try:
                    n = build_footage_index(
                        footage_root=footage_root or None,
                        output_dir="gemini_output",
                    )
                    st.success(f"Indexed {n} segments!")
                except Exception as e:
                    st.error(str(e))
                    st.exception(e)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input")
        
        uploaded_file = st.file_uploader(
            "Upload Voiceover Audio",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            help="Upload your voiceover audio file"
        )
        
        script_text = st.text_area(
            "Script (Optional)",
            height=200,
            placeholder="Paste your script here for reference...",
            help="Optional: Provide the script for better alignment"
        )
        
        process_button = st.button(
            "🚀 Process Audio",
            type="primary",
            disabled=uploaded_file is None or st.session_state.processing,
            use_container_width=True
        )
    
    with col2:
        st.header("📊 Output")
        
        if process_button and uploaded_file:
            st.session_state.processing = True
            
            try:
                temp_dir = Path("models")
                temp_dir.mkdir(exist_ok=True)
                audio_path = temp_dir / uploaded_file.name
                
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Processing..."):
                    result = asyncio.run(process_pipeline(
                        str(audio_path),
                        script_text if script_text else None
                    ))
                
                st.session_state.processed_data = result
                st.success("✅ Processing complete!")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
            finally:
                st.session_state.processing = False
        
        if st.session_state.processed_data:
            data = st.session_state.processed_data
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Video Length", f"{data['video_length']:.1f}s")
            with metrics_col2:
                st.metric("Segments", len(data['segments']))
            
            st.divider()

            display_brain_status()
            st.divider()

            display_segments_with_alternatives(data)
            st.divider()
            
            json_str = json.dumps(data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="video_segments.json",
                mime="application/json",
                use_container_width=True
            )
            
            with st.expander("📄 JSON Preview"):
                st.json(data)


if __name__ == "__main__":
    main()