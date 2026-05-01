import streamlit as st
import os
import json
import asyncio
from pathlib import Path
from typing import Optional

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

    st.info("Step 2.5/5: Microsegmentation per sentence (subtitles)...")
    micro_tasks = [
        insert_markers(seg["text"], partitions=2, model="claude-haiku-4-5")
        for seg in segments
    ]
    micro_marked_list = await asyncio.gather(*micro_tasks) if micro_tasks else []
    enriched_segments = []
    for seg, micro_marked in zip(segments, micro_marked_list):
        subtitle_lines = extract_blocks(micro_marked)
        enriched_segments.append(
            {
                **seg,
                "section": None,
                "subtitle": subtitle_lines,
            }
        )
    segments = enriched_segments
    
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


def main():
    """Main Streamlit application."""
    st.title("🎬 AI Video Editing Tool")
   
    init_models()

    with st.sidebar:
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