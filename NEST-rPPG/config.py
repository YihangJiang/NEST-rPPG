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

# ===== SWITCH BLOCK: choose ONE root layout =====
# Comment out the block you are NOT using.
# - Option A: STMap_my (PURE_my, UBFC_my) — data at REPO_ROOT/STMap_my/ or BASE_DIR/STMap_my/
# - Option B: STMap (PURE, UBFC, etc.) — data at BASE_DIR/STMap/

# --- Option A: STMap_my (uncomment and comment Option B to use PURE_my / UBFC_my) ---
# STMAP_PARENT_ROOT = REPO_ROOT   # use REPO_ROOT if STMap_my is at repo root; else BASE_DIR
# STMAP_DATA_ROOT = os.path.join(STMAP_PARENT_ROOT, 'STMap_my')
# STMAP_DATA_ROOT_REL = '../'
# STMAP_INDEX_BASE = os.path.join(BASE_DIR, 'STMap_my', 'STMap_Index')

# --- Option B: STMap ---
STMAP_PARENT_ROOT = BASE_DIR
STMAP_DATA_ROOT = os.path.join(BASE_DIR, 'STMap')
STMAP_DATA_ROOT_REL = './'
STMAP_INDEX_BASE = os.path.join(BASE_DIR, 'STMap', 'STMap_Index')

# ===== END SWITCH BLOCK =====

# Output dirs (under NEST-rPPG)
RESULT_DIR = os.path.join(BASE_DIR, 'Result')
RESULT_LOG_DIR = os.path.join(BASE_DIR, 'Result_log')
WAVE_SORT_ROOT = os.path.join(BASE_DIR, 'Wave_sort')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# ---------- Domain / run config (train + dataSort) ----------
# For STMap: use PURE, UBFC, etc. For STMap_my: use PURE_my, UBFC_my (and enable Option A above).
TGT_DOMAIN = 'PURE'      # e.g. PURE_my, PURE, UBFC_my, UBFC
SRC_DOMAIN = 'UBFC'      # single source; omit/None = use all TARGET_DOMAIN[tgt]
SPATIAL_AUG_RATE = 0.5
TEMPORAL_AUG_RATE = 0.1
LOSS_TYPE = 'All'         # One / TA / CM / DM / All

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
    'UBFC_my_in': ['STMap_my/UBFC_my_in', 'UBFC_my_in', 'STMap_RGB'],
    # Row-analysis variant for PURE (still under STMap/ by design)
    'PURE_trans_row0': ['STMap/PURE_trans_row0', 'PURE_trans_row0', 'STMap'],
}

# train_my uses domains above + TARGET_DOMAIN below to pick 3 source regions + 1 test region
STMAP_NAME = 'STMap_RGB.png'
EXP_NAME = 'PURE_my_region_align'


def get_index_dir(domain: str) -> str:
    """Index dir for a domain (e.g. PURE_my, UBFC_my) under STMAP_INDEX_BASE."""
    return os.path.join(STMAP_INDEX_BASE, domain)


def build_run_name(tgt=None, src=None, spatial=None, temporal=None, loss_type=None, override=None):
    """Build rPPGNet run name. Uses config defaults for None args."""
    if override:
        return override
    tgt = tgt or TGT_DOMAIN
    src = src or SRC_DOMAIN
    spatial = spatial if spatial is not None else SPATIAL_AUG_RATE
    temporal = temporal if temporal is not None else TEMPORAL_AUG_RATE
    loss_type = loss_type or LOSS_TYPE
    return (
        f"rPPGNet_{tgt}_src{src}"
        f"Spatial{spatial}Temporal{temporal}"
        f"_loss{loss_type}"
    )


def canonical_data_name(domain: str) -> str:
    """Map region-level domain to base dataset name used by MyDataset."""
    if domain.startswith('PURE_my'):
        return 'PURE_my'
    if domain.startswith('UBFC_my'):
        return 'UBFC_my'
    return domain
