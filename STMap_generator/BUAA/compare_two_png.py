# %%
"""
Compare two PNG images given absolute paths.
Run cells in order in Jupyter or VS Code.
Edit IMG_A and IMG_B below before running.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %%
# Config: set absolute paths to the two PNGs you want to compare
IMG_A = '/home/yj167/Desktop/NEST-rPPG/NEST-rPPG/STMap_my/lux 100.0/STMap_RGB.png'   # TODO: replace with your path
IMG_B = '/home/yj167/Desktop/NEST-rPPG/NEST-rPPG/STMap/BUAA/Sub_13lux 100.0/STMap/STMap_RGB.png'  # TODO: replace with your path

# Optional: where to save a side-by-side and diff visualization; set to None to skip
OUT_VIS = None  # e.g. '/home/you/compare_result.png'

# %%
def compare_two_images(path_a, path_b):
    """
    Load two images, resize second to first if needed,
    and compute MSE, MAE, and max absolute difference.
    """
    a = cv2.imread(path_a)
    b = cv2.imread(path_b)
    if a is None:
        raise FileNotFoundError(f'Could not load image A: {path_a}')
    if b is None:
        raise FileNotFoundError(f'Could not load image B: {path_b}')

    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    diff = np.abs(a_f - b_f)

    mse = np.mean((a_f - b_f) ** 2)
    mae = np.mean(diff)
    max_diff = np.max(diff)
    h, w = a.shape[:2]
    return a, b, mse, mae, max_diff, h, w


# %%
# Run comparison
print('Comparing:')
print('  A =', IMG_A)
print('  B =', IMG_B)

a_img, b_img, mse, mae, max_diff, h, w = compare_two_images(IMG_A, IMG_B)

print('Result:')
print(f'  size: {w}x{h}')
print(f'  MSE : {mse:.4f}')
print(f'  MAE : {mae:.4f}')
print(f'  max diff (per channel): {max_diff:.4f}')

if OUT_VIS is not None:
    # Build visualization: [A | B | amplified abs-diff]
    diff_vis = np.abs(a_img.astype(np.int32) - b_img.astype(np.int32))
    diff_vis = np.clip(diff_vis * 3, 0, 255).astype(np.uint8)  # amplify for visibility
    side = np.hstack([a_img, b_img, diff_vis])
    cv2.imwrite(OUT_VIS, side)
    print('Visualization saved to:', OUT_VIS)

# %%
# Plot pixel distribution of both images (per channel: B, G, R)


fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
channel_names = ['Blue', 'Green', 'Red']
bins = np.linspace(0, 256, 65)  # 0-255 in steps of 4

for c, (ax, name) in enumerate(zip(axes, channel_names)):
    vals_a = a_img[:, :, c].ravel()
    vals_b = b_img[:, :, c].ravel()
    # Use outlined histograms so the two curves don't overlap into brown
    ax.hist(vals_a, bins=bins, label='Image A', color='#1f77b4', density=True, histtype='step', linewidth=2)
    ax.hist(vals_b, bins=bins, label='Image B', color='#ff7f0e', density=True, histtype='step', linewidth=2)
    ax.set_ylabel('Density')
    ax.set_title(f'Channel: {name}')
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Pixel value (0–255)')
fig.suptitle('Pixel distribution: Image A vs Image B')
plt.tight_layout()
plt.show()

# %%
