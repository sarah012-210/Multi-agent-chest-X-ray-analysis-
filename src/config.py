"""
Central configuration for the CheXpert multi-agent pipeline.
Edit the paths below to match your machine after running data/load.py.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Paste the path printed by `data/load.py` (kagglehub.dataset_download output)
# here if your images live in the kagglehub cache instead of DATA_DIR.
KAGGLEHUB_PATH = os.environ.get("CHEXPERT_KAGGLEHUB_PATH", os.path.join(DATA_DIR, "chexpert"))

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VALID_CSV = os.path.join(DATA_DIR, "valid.csv")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
IMAGE_AGENT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "image_agent.pt")

KNOWLEDGE_BASE_JSON = os.path.join(
    PROJECT_ROOT, "src", "knowledge", "knowledge_base.json"
)
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "knowledge.index")

# ---------------------------------------------------------------------------
# CheXpert labels
# ---------------------------------------------------------------------------
# The 14 observation columns in train.csv / valid.csv (order matters — it is
# the order used everywhere else in this project).
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# The 5 "competition" labels most CheXpert papers report on. Kept separate
# because you may want to train/evaluate on just these for a faster project.
COMPETITION_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

CHEXPERT_LABELS = COMPETITION_LABELS

NUM_CLASSES = len(CHEXPERT_LABELS)

# How to handle the "-1" (uncertain) label in the CSVs.
# "ones"  -> treat uncertain as positive (1)   [common baseline, u-ones]
# "zeros" -> treat uncertain as negative (0)   [common baseline, u-zeros]
# "ignore" -> drop rows containing -1 for that label during loss computation
UNCERTAIN_POLICY = "ones"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda"  # falls back to cpu automatically in image_agent.py if unavailable

# ---------------------------------------------------------------------------
# RAG / LLM
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PREDICTION_THRESHOLD = 0.5  # probability above which a condition is "flagged"
