#!/usr/bin/env python3
"""Check that cr_common.encode_multi reproduces s03.fit_encoding_model column by column."""

from cr_common import (init_run, load_brain, load_features, lagged_pca, encode_multi,
                       brain_matrix, unpack, ROI_NAMES)
import numpy as np
from scripts.s03_encoding_models import fit_encoding_model

init_run()
subjects, brain, n_trs = load_brain()
X, rows = lagged_pca(load_features('GPT-2-Medium', 'shapessocial', 9, n_trs))
r_multi = unpack(encode_multi(X, brain_matrix(brain['shapessocial'], rows)), len(subjects))
diffs = []
for si in range(5):
    for ri in range(len(ROI_NAMES)):
        r_single, _ = fit_encoding_model(X, brain['shapessocial'][si, rows, ri])
        diffs.append(abs(r_single - r_multi[si, ri]))
print(f'max |r_single - r_multi| over 30 columns: {max(diffs):.2e}')
