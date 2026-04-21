import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
KEYS_ENV_PATH = BASE_DIR / 'data' / 'keys.env'
if KEYS_ENV_PATH.exists():
    load_dotenv(KEYS_ENV_PATH)

REGISTRY_DIR = BASE_DIR / 'data' / 'registry'
BENCHMARKS_DIR = BASE_DIR / 'data' / 'benchmarks'

# Public registry keeps JSON examples only.
INSTRUMENTS_EN_PATH = REGISTRY_DIR / 'instruments_en.json'
CONSUMABLES_EN_PATH = REGISTRY_DIR / 'consumables_en.json'
INSTRUMENTS_PATH = INSTRUMENTS_EN_PATH
CONSUMABLES_PATH = CONSUMABLES_EN_PATH

# Retrieval assets are intentionally not shipped in public release.
DATABASE_DIR = Path(os.getenv('DATABASE_DIR', ''))
FAISS_INDEX_PATH = Path(os.getenv('FAISS_INDEX_PATH', ''))
CHUNKS_JSON_PATH = Path(os.getenv('CHUNKS_JSON_PATH', ''))

TEST_SETS_DIR = BENCHMARKS_DIR
SUBSET_A_PATH = TEST_SETS_DIR / 'ProAgent-SubsetA.json'
SUBSET_B_PATH = TEST_SETS_DIR / 'ProAgent-SubsetB.json'
SUBSET_C_PATH = TEST_SETS_DIR / 'ProAgent-SubsetC.json'
SUBSET_D_PATH = TEST_SETS_DIR / 'ProAgent-SubsetD.json'

OUTPUTS_DIR = BASE_DIR / 'outputs'
LOGS_DIR = OUTPUTS_DIR / 'logs'
RESULTS_DIR = OUTPUTS_DIR / 'results'
EVAL_RESULTS_DIR = OUTPUTS_DIR / 'evaluation'
for p in [OUTPUTS_DIR, LOGS_DIR, RESULTS_DIR, EVAL_RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv('MODEL_NAME', 'qwen3-30b-a3b')
MODEL_API_KEY = os.getenv('MODEL_API_KEY', '')
MODEL_BASE_URL = os.getenv('MODEL_BASE_URL', '')
MODEL_TEMPERATURE = float(os.getenv('MODEL_TEMPERATURE', '0'))
MODEL_MAX_TOKENS = int(os.getenv('MODEL_MAX_TOKENS', '8000'))

# Role models are optional user extensions; default to the same endpoint.
FAST_MODEL_NAME = os.getenv('FAST_MODEL_NAME', MODEL_NAME)
GENERAL_MODEL_NAME = os.getenv('GENERAL_MODEL_NAME', MODEL_NAME)
QUALITY_MODEL_NAME = os.getenv('QUALITY_MODEL_NAME', MODEL_NAME)
LONG_CONTEXT_MODEL_NAME = os.getenv('LONG_CONTEXT_MODEL_NAME', MODEL_NAME)

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', '')
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY', '')
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', '')

MEMORY_MODEL_NAME = os.getenv('MEMORY_MODEL_NAME', MODEL_NAME)

EVAL_CONFIG = {
    'max_retries': 3,
    'timeout_seconds': 120,
    'save_intermediate': True,
    'verbose': True,
}

# Public shell: one profile only (22-device concept retained, details hidden).
RULE_PROFILE = 'public22_shell'

# Keep IDs configurable without exposing private constraints.
VALID_INSTRUMENT_IDS = set()
VALID_CONSUMABLE_IDS = set()
