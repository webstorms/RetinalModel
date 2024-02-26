import sys
sys.path.append("build_pc.py")
sys.path.append("fit_readout.py")

import build_pc
import fit_readout

build_pc.build_pcs(dataset_name="MouseMovie", spatial_args=[(0.4, 80.0)])
fit_readout.fit_readouts(n_epochs=2000, batch_size=8, lambdas=[10**-5, 10**-5.5, 10**-6, 10**-6.5, 10**-7], dataset_name="MouseMovie", spatial_args=[(0.4, 80.0)], rf_len_ms=46)
