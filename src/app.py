"""
Streamlit UI for the CheXpert multi-agent pipeline.

Run with:
    streamlit run src/app.py
"""
import os
import sys
import tempfile

# Streamlit sets sys.path to this file's own folder (src/), not the project
# root, so "from src.config import ..." fails unless we add the root manually.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.config import PREDICTION_THRESHOLD
from src.pipeline import CheXpertPipeline
from src.report.pdf_generator import generate_pdf_report

st.set_page_config(
    page_title="CheXpert AI Diagnostic Assistant",
    page_icon="🫁",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.2rem;
    }
    .app-header h1 {
        font-size: 1.9rem;
        margin: 0;
        color: #1F4E79;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.4rem;
    }
    .disclaimer-banner {
        background: #FFF8E6;
        border: 1px solid #F2D06B;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        color: #7A5A00;
        margin-bottom: 1.5rem;
    }
    /* This targets Streamlit's real bordered-container wrapper element,
       so the border actually surrounds the title + content together
       (raw <div> markdown tags don't nest content in Streamlit). */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid   !important;
        border-radius: 12px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 12px !important;
    }
    .section-card h3, .section-title {
        margin-top: 0;
        font-size: 1.05rem;
        color: #1F4E79;
    }
    .stButton > button {
        background-color: #1F4E79;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #163a5c;
        color: white;
    }
    .stDownloadButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <span style="font-size: 2rem;">🫁</span>
    <h1>CheXpert AI Diagnostic Assistant</h1>
</div>
<div class="app-subtitle">Multi-agent chest X-ray analysis — Image Agent · RAG Agent · Report Agent · Chat Agent</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <b>Research/educational project — not a diagnostic medical device.</b>
    All output is an AI-generated draft requiring review by a qualified radiologist or physician.
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return CheXpertPipeline()


if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

with st.spinner("Loading agents (Image, RAG, Report)... this happens once."):
    pipeline = get_pipeline()

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    with st.container(border=True):
        st.markdown("### 1 · Upload X-ray")
        uploaded_file = st.file_uploader("Chest X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        use_custom_threshold = st.checkbox(
            "Override with one flat threshold for all labels",
            value=False,
            help="By default each condition uses its own F1-optimal threshold "
                 "found during training, since conditions like Cardiomegaly and "
                 "Atelectasis need very different cutoffs."
        )
        threshold = None
        if use_custom_threshold:
            threshold = st.slider("Flagging threshold (applied to all labels)", 0.1, 0.9, PREDICTION_THRESHOLD, 0.05)

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)

            if st.button("▶  Run Analysis", type="primary", use_container_width=True):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Running Image Agent → RAG Agent → Report Agent..."):
                    st.session_state.result = pipeline.run(tmp_path, threshold=threshold)
                    st.session_state.chat_history = []
                    st.session_state.image_path = tmp_path
                    st.session_state.uploaded_name = uploaded_file.name

with col2:
    with st.container(border=True):
        st.markdown("### 2 · Predictions & Report")

        if st.session_state.result:
            result = st.session_state.result

            st.markdown("**Predicted probabilities**")
            sorted_preds = sorted(result["predictions"].items(), key=lambda x: -x[1])
            for cond, prob in sorted_preds:
                flagged = cond in result["flagged_conditions"]
                label = f"{'🔴' if flagged else '⚪'} {cond}: {prob:.2f}"
                st.progress(min(prob, 1.0), text=label)

            st.markdown('<hr style="border: none; border-top: 1px solid #7C3AED; margin: 1rem 0;">', unsafe_allow_html=True)
            st.markdown("**Clinical Report (AI Draft)**")
            st.markdown(result["report"])

            # --- PDF download ---
            pdf_bytes = generate_pdf_report(
                image_path=st.session_state.image_path,
                predictions=result["predictions"],
                report_text=result["report"],
                patient_label=st.session_state.uploaded_name or "uploaded_xray",
            )
            st.download_button(
                label="⬇  Download Report as PDF",
                data=pdf_bytes,
                file_name="chexpert_ai_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("Upload an X-ray and click **Run Analysis** to see predictions and a report here.")

with st.container(border=True):
    st.markdown("### 3 · Ask questions about the report")

    if st.session_state.result:
        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])

        question = st.chat_input("Ask a question about the findings...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = pipeline.chat(question)
                    st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.caption("Run an analysis first to unlock the chat.")