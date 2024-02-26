import sys
sys.path.append("build_pc.py")
sys.path.append("../search/fit_readout.py")
from scripts.fits.search import build_pc, fit_readout

spatial_args = [(0.078125, 20.0), (0.1171875, 30.0), (0.1796875, 46.0)]
build_pc.build_pcs(dataset_name="SalamanderImage", spatial_args=spatial_args)
fit_readout.fit_readouts(n_epochs=2000, batch_size=32, lambdas=[10**-3.5, 10**-4, 10**-4.5, 10**-5, 10**-5.5], dataset_name="SalamanderImage", spatial_args=spatial_args)
