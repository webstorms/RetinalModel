import sys
sys.path.append("build_pc.py")
sys.path.append("../search/fit_readout.py")
from scripts.fits.search import build_pc, fit_readout

# First one is 12px and second one is 16px
spatial_args = [(0.122, 20.0), (0.19634, 20.0), (0.2927, 24.0)]
build_pc.build_pcs(dataset_name="MacaqueImage", spatial_args=spatial_args)
fit_readout.fit_readouts(n_epochs=2000, batch_size=8, lambdas=[10 ** -2, 10 ** -2.5, 10 ** -3, 10 ** -3.5, 10 ** -4], dataset_name="MacaqueImage", spatial_args=spatial_args)
