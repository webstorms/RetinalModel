import sys
sys.path.append("build_pc.py")
sys.path.append("fit_readout.py")

import build_pc
import fit_readout

# build_pc.build_pcs(dataset_name="SalamanderImage", spatial_args=[(0.1171875, 30.0)])
fit_readout.fit_readouts(n_epochs=2000, batch_size=32, lambdas=[10**-3.5, 10**-4, 10**-4.5, 10**-5, 10**-5.5], dataset_name="SalamanderImage", spatial_args=[(0.1171875, 30.0)])
