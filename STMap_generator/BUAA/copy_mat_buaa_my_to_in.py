# %%
"""
Copy all .mat files from STMap_my/BUAA_my to STMap_my/BUAA_my_in,
preserving the same folder structure (e.g. Sub_01lux 10.0/Label/BVP.mat -> same path under BUAA_my_in).
Run from repo root or from this directory; paths are relative to the script location.
"""
import os
import shutil

# %%
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..'))

BUAA_MY_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'BUAA_my')
BUAA_MY_IN_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'BUAA_my_rm')

# %%
if not os.path.isdir(BUAA_MY_ROOT):
    print('BUAA_my root not found:', BUAA_MY_ROOT)
else:
    os.makedirs(BUAA_MY_IN_ROOT, exist_ok=True)
    copied = 0
    for dirpath, _dirnames, filenames in os.walk(BUAA_MY_ROOT):
        rel = os.path.relpath(dirpath, BUAA_MY_ROOT)
        for f in filenames:
            if not f.lower().endswith('.mat'):
                continue
            src = os.path.join(dirpath, f)
            dest_dir = os.path.join(BUAA_MY_IN_ROOT, rel)
            dest = os.path.join(dest_dir, f)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
            print('  ', rel, f)
    print('Done. Copied', copied, '.mat files to', BUAA_MY_IN_ROOT)

# %%
