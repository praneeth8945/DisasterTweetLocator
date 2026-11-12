"""Central configuration for DisasterTweetLocator.

Values stated in the paper are marked [paper]. Values taken from the
original GLOVE_LSTM notebook are marked [notebook]. Others are [chosen].
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# Dataset: Kaggle "Natural Language Processing with Disaster Tweets"
# columns: id, keyword, location, text, target            [paper, Sec. III-A]
TRAIN_CSV = DATA_DIR / "train.csv"   # 7,613 labelled tweets [paper]
TEST_CSV = DATA_DIR / "test.csv"     # unlabelled Kaggle test set (used only for demo predictions)

# GloVe embeddings
GLOVE_PATH = DATA_DIR / "glove.6B.50d.txt"   # [paper]
EMBEDDING_DIM = 50                            # [paper]

# Text / sequence
MAX_VOCAB = 10000        # [notebook]
MAX_LEN = 50             # [notebook]

# Model
RNN_UNITS = 128          # [notebook]
DENSE_UNITS = 64         # [notebook]
DROPOUT = 0.5            # [notebook]
TRAINABLE_EMBEDDINGS = False   # [notebook] frozen GloVe

# Training
LEARNING_RATE = 0.001    # [paper, Sec. IV-B]  Adam optimiser
LOSS = "binary_crossentropy"   # [paper]
BATCH_SIZE = 64          # [notebook]
EPOCHS = 15              # [chosen] upper bound; early stopping picks the epoch
EARLY_STOP_PATIENCE = 3  # [chosen]

# Evaluation protocol  [chosen]
#   train.csv -> 20% held-out TEST (fixed, random_state=42), never used for training or selection
#             -> remaining 80% split 87.5 / 12.5 into TRAIN / VALIDATION (per seed)
#   VALIDATION drives early stopping; reported numbers are on TEST.
TEST_SIZE = 0.2
TEST_SPLIT_SEED = 42
VAL_SIZE = 0.125
SEEDS = [42, 1, 2, 3, 4]  # every model is trained once per seed; mean ± sd reported
MAIN_SEED = 42            # seed whose model/curves are saved for the demo + paper figures

# Geolocation
SPACY_MODEL = "en_core_web_sm"                 # [paper, Sec. III-E]
GEOCODER_USER_AGENT = "disaster_tweet_locator"
GEOCODE_DELAY_SEC = 1.0  # Nominatim usage policy: max 1 request / second
GEOCODE_CACHE = DATA_DIR / "geocode_cache.json"   # offline fallback for the demo tweets
