import sys
sys.path.append("build_pc.py")
sys.path.append("fit_readout.py")

import build_pc
import fit_readout

build_pc.build_pcs(dataset_name="MacaqueImage", spatial_args=[(0.19634, 20.0)])
fit_readout.fit_readouts(n_epochs=2000, batch_size=8, lambdas=[10**-2, 10**-2.5, 10**-3, 10**-3.5, 10**-4], dataset_name="MacaqueImage", spatial_args=[(0.19634, 20.0)])
