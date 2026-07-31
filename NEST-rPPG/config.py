# -*- coding: UTF-8 -*-
"""
Shared config for train.py, train_my.py, and dataSort.py.
Edit this file to change paths and key run parameters in one place.
"""
import os

# ---------- Paths ----------
# Directory containing this config (NEST-rPPG subfolder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root (one level up from NEST-rPPG)
REPO_ROOT = os.path.dirname(BASE_DIR)

# ===== SWITCH BLOCK: choose ONE dataset layout =====
# Both STMap and STMap_my now live under REPO_ROOT, so we only switch the folder name.
# Comment out the block you are NOT using.

# --- Option A: Use original STMap (PURE, UBFC, BUAA, VIPL, V4V) at REPO_ROOT/STMap/... ---
# STMAP_PARENT_ROOT = REPO_ROOT
# STMAP_DATA_ROOT = os.path.join(STMAP_PARENT_ROOT, 'STMap')
# STMAP_DATA_ROOT_REL = '../'
# STMAP_INDEX_BASE = os.path.join(BASE_DIR, 'STMap', 'STMap_Index')

# --- Option B: Use STMap_my (PURE_my, UBFC_my, BUAA_my, regions) at REPO_ROOT/STMap_my/... ---
STMAP_PARENT_ROOT = REPO_ROOT
STMAP_DATA_ROOT = os.path.join(STMAP_PARENT_ROOT, 'STMap_my')
STMAP_DATA_ROOT_REL = '../'
STMAP_INDEX_BASE = os.path.join(BASE_DIR, 'STMap_my', 'STMap_Index')

# ===== END SWITCH BLOCK =====

# Output dirs (under NEST-rPPG)
RESULT_DIR = os.path.join(BASE_DIR, 'Output')
RESULT_LOG_DIR = os.path.join(BASE_DIR, 'Training_Log')
WAVE_SORT_ROOT = os.path.join(BASE_DIR, 'Wave_sort')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# MLflow (override via MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME env vars)
# MLflow 3.3+ requires a database backend (not file://). Artifacts still go under mlruns/.
MLFLOW_ARTIFACT_ROOT = os.path.join(RESULT_LOG_DIR, 'mlruns')
MLFLOW_DB_PATH = os.path.join(RESULT_LOG_DIR, 'mlflow.db')
MLFLOW_TRACKING_URI = os.environ.get(
    'MLFLOW_TRACKING_URI',
    'sqlite:///' + os.path.abspath(MLFLOW_DB_PATH),
)
MLFLOW_EXPERIMENT_NAME = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'nest-rppg')

# ---------- Domain / run config (train + dataSort) ----------
# For STMap: use PURE, UBFC, etc. For STMap_my: use PURE_my, UBFC_my (and enable Option A above).
TGT_DOMAIN = 'BUAA_my_in'      # e.g. PURE_my, PURE, UBFC_my, UBFC
SRC_DOMAIN = 'PURE_my_in'      # single source; omit/None = use all TARGET_DOMAIN[tgt]
SPATIAL_AUG_RATE = 0.5
TEMPORAL_AUG_RATE = 0.1
LOSS_TYPE = 'One'         # One / TA / CM / DM / All
WEIGHT_INFO = 0.0        # InfoNCE alignment weight (0 = disabled)
WEIGHT_INFO_SWEEP = (0.01,)  # weight_info fixed to 0.01 so Optuna sweeps tau_info only
# Optuna search space (optuna_tune_regions.py)
OPTUNA_TAU_INFO_SWEEP = (0.01, 0.05, 0.1, 0.5)  # tau_info candidates (0.01, 0.05, 0.1, 0.5)
SEED = 0

# Mapping from target domain to list of all possible source domains
TARGET_DOMAIN = {
    'VIPL': ['V4V', 'PURE', 'BUAA', 'UBFC'],
    'V4V': ['VIPL', 'PURE', 'BUAA', 'UBFC'],
    'PURE': ['VIPL', 'V4V', 'BUAA', 'UBFC'],
    'BUAA': ['VIPL', 'V4V', 'PURE', 'UBFC'],
    'UBFC': ['VIPL', 'V4V', 'PURE', 'BUAA'],
    # STMap_my variants (use _my sources when STMap/ not present)
    'PURE_my': ['BUAA_my', 'UBFC_my'],
    'UBFC_my': ['PURE_my', 'BUAA_my'],
    # train_my: test domain -> 3 source-region domains (order matters: cheek, target, eye)
    'UBFC_my_in': ['PURE_my_rm', 'PURE_my_in', 'PURE_my_eye'],
    'PURE_my_in': ['BUAA_my_rm', 'BUAA_my_in', 'BUAA_my_eye'],
    'BUAA_my_in': ['PURE_my_rm', 'PURE_my_rm', 'PURE_my_rm'],
    # 'BUAA_my_rm': ['UBFC_my_rm', 'UBFC_my_rm', 'UBFC_my_rm'],
    'BUAA_my_eye': ['UBFC_my_eye', 'UBFC_my_eye', 'UBFC_my_eye'],

}

# Data paths: PURE and UBFC use STMap_my (PURE_my, UBFC_my); others use STMap/
FILEA_NAME = {
    # Original NEST-rPPG STMap folders
    'VIPL': ['STMap/VIPL', 'VIPL', 'STMap_RGB_Align_CSI'],
    'V4V': ['STMap/V4V', 'V4V', 'STMap_RGB'],
    'PURE': ['STMap/PURE', 'PURE', 'STMap'],
    'BUAA': ['STMap/BUAA', 'BUAA', 'STMap_RGB'],
    'UBFC': ['STMap/UBFC', 'UBFC', 'STMap'],
    # STMap_my variants (PURE_my, UBFC_my, BUAA_my)
    'PURE_my': ['STMap_my/PURE_my', 'PURE_my', 'STMap_RGB'],
    'UBFC_my': ['STMap_my/UBFC_my', 'UBFC_my', 'STMap_RGB'],
    'BUAA_my': ['STMap_my/BUAA_my', 'BUAA_my', 'STMap_RGB'],
    # Region / subfolder variants used by train_my.py
    'PURE_my_rm': ['STMap_my/PURE_my_rm', 'PURE_my_rm', 'STMap_RGB'],
    'PURE_my_in': ['STMap_my/PURE_my_in', 'PURE_my_in', 'STMap_RGB'],
    'PURE_my_eye': ['STMap_my/PURE_my_eye', 'PURE_my_eye', 'STMap_RGB'],
    'UBFC_my_rm': ['STMap_my/UBFC_my_rm', 'UBFC_my_rm', 'STMap_RGB'],
    'UBFC_my_in': ['STMap_my/UBFC_my_in', 'UBFC_my_in', 'STMap_RGB'],
    'UBFC_my_eye': ['STMap_my/UBFC_my_eye', 'UBFC_my_eye', 'STMap_RGB'],
    'BUAA_my_rm': ['STMap_my/BUAA_my_rm', 'BUAA_my_rm', 'STMap_RGB'],
    'BUAA_my_in': ['STMap_my/BUAA_my_in', 'BUAA_my_in', 'STMap_RGB'],
    'BUAA_my_eye': ['STMap_my/BUAA_my_eye', 'BUAA_my_eye', 'STMap_RGB'],
    # Row-analysis variant for PURE (still under STMap/ by design)
    'PURE_trans_row0': ['STMap/PURE_trans_row0', 'PURE_trans_row0', 'STMap'],
}

# train_my uses domains above + TARGET_DOMAIN below to pick 3 source regions + 1 test region
STMAP_NAME = 'STMap_RGB.png'
EXP_NAME = 'PURE_my_region_align'


def get_index_dir(domain: str) -> str:
    """Index dir for a domain (e.g. PURE_my, UBFC_my) under STMAP_INDEX_BASE."""
    return os.path.join(STMAP_INDEX_BASE, domain)


def build_run_name(
    tgt=None,
    src=None,
    weight_info=None,
):
    """
    Build rPPGNet run name: rPPGNet_<tgt>_src<src>_w<weight_info>.
    """
    tgt = tgt or TGT_DOMAIN
    src = src or SRC_DOMAIN
    w = float(WEIGHT_INFO if weight_info is None else weight_info)
    return f"rPPGNet_{tgt}_src{src}_w{'%g' % w}"


def canonical_data_name(domain: str) -> str:
    """Map region-level domain to base dataset name used by MyDataset."""
    if domain.startswith('PURE_my'):
        return 'PURE_my'
    if domain.startswith('UBFC_my'):
        return 'UBFC_my'
    return domain


# ===== EVAL_SAVE_PATH: choose ONE for eval_from_bvp =====
# Comment out the option you are NOT using.

# --- Option A: train.py output (WAVE_SORT_ROOT / TGT_DOMAIN / build_run_name()) ---
EVAL_SAVE_PATH = os.path.join(WAVE_SORT_ROOT, TGT_DOMAIN, build_run_name(weight_info=WEIGHT_INFO))

# --- Option B: train_regions output (set test_domain and run name to match your regions run) ---
# EVAL_SAVE_PATH = os.path.join(WAVE_SORT_ROOT, 'UBFC_my_in', 'rPPGNet_UBFC_my_in_srcPURE_my_in')

# ===== END EVAL_SAVE_PATH =====
