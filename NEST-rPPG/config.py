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

# STMap data: at repo root (STMap_my/PURE_my, STMap_my/UBFC_my, etc.)
STMAP_DATA_ROOT = os.path.join(REPO_ROOT, 'STMap_my')
# Relative path from NEST-rPPG for scripts that use root_file
STMAP_DATA_ROOT_REL = '../'

# Centralized STMap indexes: NEST-rPPG/STMap_my/STMap_Index/<domain>
STMAP_INDEX_BASE = os.path.join(BASE_DIR, 'STMap_my', 'STMap_Index')

# For train_my region-aware index (separate layout)
STMAP_INDEX_MY_REGIONS = os.path.join(BASE_DIR, 'STMap_my', 'STMap_Index_my_regions')

# Output dirs (under NEST-rPPG)
RESULT_DIR = os.path.join(BASE_DIR, 'Result')
RESULT_LOG_DIR = os.path.join(BASE_DIR, 'Result_log')
WAVE_SORT_ROOT = os.path.join(BASE_DIR, 'Wave_sort')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# ---------- Domain / run config (train + dataSort) ----------
TGT_DOMAIN = 'PURE_my'   # e.g. PURE_my, PURE, UBFC_my, UBFC
SRC_DOMAIN = 'UBFC_my'   # e.g. UBFC or UBFC_my (must match args.src; use UBFC_my if STMap/UBFC not present)
SPATIAL_AUG_RATE = 0.5
TEMPORAL_AUG_RATE = 0.1
LOSS_TYPE = 'One'        # One / TA / CM / DM / All

# Mapping from target domain to list of possible source domains
TARGET_DOMAIN = {
    'VIPL': ['V4V', 'PURE', 'BUAA', 'UBFC'],
    'V4V': ['VIPL', 'PURE', 'BUAA', 'UBFC'],
    'PURE': ['VIPL', 'V4V', 'BUAA', 'UBFC'],
    'BUAA': ['VIPL', 'V4V', 'PURE', 'UBFC'],
    'UBFC': ['VIPL', 'V4V', 'PURE', 'BUAA'],
    # STMap_my variants (use _my sources when STMap/ not present)
    'PURE_my': ['V4V', 'PURE', 'BUAA_my', 'UBFC_my'],
    'UBFC_my': ['VIPL', 'V4V', 'PURE', 'BUAA_my'],
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
    # Row-analysis variant for PURE (still under STMap/ by design)
    'PURE_trans_row0': ['STMap/PURE_trans_row0', 'PURE_trans_row0', 'STMap'],
}

# ---------- train_my region config ----------
CHEEK_ROOT = 'STMap_my/PURE_my_rm'
TARGET_ROOT = 'STMap_my/PURE_my_in'
EYE_ROOT = 'STMap_my/PURE_my_eye'
UBFC_MY_ROOT = 'STMap_my/UBFC_my'
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
