# P(p)=C(p)⋅D(p)

import numpy as np

# Confidence map
C = np.ones_like(mask, dtype=np.float32)
C[mask == 255] = 0   # zone blanche = inconnue = confiance 0

P = C.copy()   # mtn on met d=1 pr simplifier
