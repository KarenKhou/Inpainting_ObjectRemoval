import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Charger l'image avec alpha ---
image_rgba = cv2.imread("image_incomplete.png", cv2.IMREAD_UNCHANGED)
alpha = image_rgba[:, :, 3]

# mask de la zone a remplir
mask_target = np.zeros_like(alpha, dtype=np.uint8)
mask_target[alpha == 0] = 1 


image_bgr = image_rgba[:, :, :3].copy()
image_bgr[mask_target == 1] = [0, 0, 0]  # efface visuellement


patch_radius = 15
h, w = mask_target.shape
C = 1 - mask_target.astype(np.float32)  # carte de confiance initiale

def compute_confidence(mask, C, contours):
    """Calcule la confiance moyenne dans un patch autour de chaque point du contour"""
    patch_r = patch_radius
    h, w = mask.shape
    contour_xy, C_values = [], []
    for cnt in contours:
        for pt in cnt:
            x, y = int(pt[0][0]), int(pt[0][1])
            x0, x1 = max(0, x - patch_r), min(w, x + patch_r + 1)
            y0, y1 = max(0, y - patch_r), min(h, y + patch_r + 1)
            patch = C[y0:y1, x0:x1]
            C_avg = float(patch.mean())
            contour_xy.append((x, y))
            C_values.append(C_avg)
    return np.array(contour_xy), np.array(C_values)

iteration = 0
max_iter = 300

while np.any(mask_target == 1) and iteration < max_iter:
    iteration += 1
    print(f"\nIteration {iteration}")

    #trouver le contour de la zone a remplir
    mask_uint8 = (mask_target * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        break

    # calcul de conf
    contour_xy, C_values = compute_confidence(mask_target, C, contours)

    # trouver le pt de confiance max
    best_idx = np.argmax(C_values)
    px, py = contour_xy[best_idx]
    print(f" Point choisi ({px},{py}), confiance = {C_values[best_idx]:.3f}")

    #  Avancer d un pixel dans le trou pr pas etre sur le contour
    offset = 1
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            yy = min(max(py + dy, 0), h - 1)
            xx = min(max(px + dx, 0), w - 1)
            if mask_target[yy, xx] == 1:
                px, py = xx, yy
                break

    # Extraire le patch cible
    x0, x1 = max(0, px - patch_radius), min(w, px + patch_radius + 1)
    y0, y1 = max(0, py - patch_radius), min(h, py + patch_radius + 1)
    target_patch = image_bgr[y0:y1, x0:x1]
    target_mask = mask_target[y0:y1, x0:x1]

    print("Pixels à remplir dans ce patch :", np.count_nonzero(target_mask == 1))
    if np.count_nonzero(target_mask == 1) == 0:
        continue

    # chercher le patch source le plus similaire
    min_ssd = 1000000000000
    best_qx, best_qy = 0, 0
    for qy in range(patch_radius, h - patch_radius):
        for qx in range(patch_radius, w - patch_radius):
            if mask_target[qy, qx] == 1:
                continue
            sx0, sx1 = qx - patch_radius, qx + patch_radius + 1
            sy0, sy1 = qy - patch_radius, qy + patch_radius + 1
            source_patch = image_bgr[sy0:sy1, sx0:sx1]
            valid = (target_mask == 0)
            if np.count_nonzero(valid) == 0:
                continue

            # s'assurer que les deux patchs ont la même taille
            th, tw = target_patch.shape[:2]
            sh, sw = source_patch.shape[:2]
            hh = min(th, sh)
            ww = min(tw, sw)
            target_patch = target_patch[:hh, :ww]
            source_patch = source_patch[:hh, :ww]
            valid = valid[:hh, :ww]

            diff = (target_patch.astype(np.float32) - source_patch.astype(np.float32)) ** 2
            valid3 = np.repeat(valid[:, :, None], 3, axis=2)
            ssd = np.sum(diff[valid3]) / np.count_nonzero(valid)
            if ssd < min_ssd:
                min_ssd = ssd
                best_qx, best_qy = qx, qy

    print(f"par ({best_qx},{best_qy}), SSD={min_ssd:.2f}")

    #  Copier les pixels manquants 
    sx0, sx1 = best_qx - patch_radius, best_qx + patch_radius + 1
    sy0, sy1 = best_qy - patch_radius, best_qy + patch_radius + 1
    source_patch = image_bgr[sy0:sy1, sx0:sx1]

    filled_image = image_bgr.copy()
    #filled_image[y0:y1, x0:x1][target_mask == 1] = [0, 0, 255]  # Debug rouge
    filled_image[y0:y1, x0:x1][target_mask == 1] = source_patch[target_mask == 1]


    # Mettre à jour image + masque + confiance
    image_bgr = filled_image.copy()
    mask_target[y0:y1, x0:x1][target_mask == 1] = 0
    C[y0:y1, x0:x1][target_mask == 1] = C_values[best_idx]

    print("Zone restante à remplir :", np.count_nonzero(mask_target))

    if iteration % 25 == 0 or np.all(mask_target == 0):
        plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        plt.title(f"Résultat après {iteration} itérations")
        plt.axis("off")
        plt.show()

print("Inpainting termine")
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.title(f"res après {iteration} iterations")
plt.axis("off")
plt.show()