import sys
sys.path.append("build_pc.py")
sys.path.append("../search/fit_readout.py")
from scripts.fits.search import build_pc, fit_readout

spatial_args = [(0.18, 36.0), (0.27, 54.0), (0.4, 80.0)]
build_pc.build_pcs(dataset_name="MouseMovie", spatial_args=spatial_args)
fit_readout.fit_readouts(n_epochs=2000, batch_size=8, lambdas=[10**-5, 10**-5.5, 10**-6, 10**-6.5, 10**-7], dataset_name="MouseMovie", spatial_args=spatial_args, rf_len_ms=46)
