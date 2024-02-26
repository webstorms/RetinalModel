import torch
import pandas as pd
import brainbox.physiology as phys

import retina.neural.train as train
import retina.neural.pca as pca


class FitMetricsBuilder:

    def __init__(self, root, dataset_name, is_train, pred_ms, spatial_scale, max_dim, luminance=1, rf_len_ms=21):
        print(f"dataset_name={dataset_name} pred_ms={pred_ms} spatial_scale={spatial_scale} max_dim={max_dim}")
        self._root = root

        self._model = self._load_model(dataset_name, pred_ms, spatial_scale, max_dim, luminance, rf_len_ms)
        self._dataset = self._load_dataset(dataset_name, is_train, spatial_scale, max_dim, luminance, pred_ms)

        # Compute outputs
        self.x, self.pred_y, self.target_y = self._compute_output(self._dataset)

        self.cc = self._compute_cc(self.pred_y, self.target_y)
        self.cc_norm = self.cc / self._dataset.cc_max

    def _load_model(self, dataset_name, pred_ms, spatial_scale, max_dim, luminance=1, rf_len_ms=13):
        fit_id = f"{dataset_name}_{pred_ms}_{spatial_scale}_{max_dim}_{luminance}_{rf_len_ms}"
        readout = train.Trainer.load_model(f"{self._root}/fits/{fit_id}/models", "0")

        return readout

    def _load_dataset(self, dataset_name, is_train, spatial_scale=0.1796875, max_dim=46, luminance=0.5, pred_ms=60):
        dataset = pca.PCDataset(self._root, train=is_train, n_pca=500, cell_idx=None, subclip_ms=None)
        dataset.load_responses(dataset=dataset_name, spatial_scale=spatial_scale, max_dim=max_dim, luminance=luminance)
        dataset.load_model(lam=10**-2.5, noise_type="*", photo_noise=0.01, ganglion_noise=0.6, pred_ms=pred_ms, decoder_span=2)

        return dataset

    def _compute_output(self, dataset):
        x = dataset._x.unsqueeze(1).cuda().float()
        target_y = dataset._y.mean(2)

        with torch.no_grad():
            pred_y = self._model(x).cpu().detach()

        t_off = target_y.shape[1] - pred_y.shape[1]
        target_y = target_y[:, t_off:, :]

        return x, pred_y, target_y

    def _compute_cc(self, pred_y, target_y):
        pred_y = pred_y.flatten(0, 1).permute(1, 0)
        target_y = target_y.flatten(0, 1).permute(1, 0)

        return phys.neural.cc(pred_y, target_y)


class FitMetricsSummary:

    def __init__(self, root, dataset_name, is_train, pred_ms_list, spatial_args_list, luminance=1, rf_len_ms=21):
        self._root = root
        self._dataset_name = dataset_name
        self._is_train = is_train
        self._pred_ms_list = pred_ms_list
        self._spatial_args_list = spatial_args_list
        self._luminance = luminance
        self._rf_len_ms = rf_len_ms

    def build_df(self):
        data_list = []

        for pred_ms in self._pred_ms_list:
            for (spatial_scale, max_dim) in self._spatial_args_list:
                try:
                    fit_metrics = FitMetricsBuilder(self._root, self._dataset_name, is_train=self._is_train, pred_ms=pred_ms, spatial_scale=spatial_scale, max_dim=max_dim, luminance=self._luminance, rf_len_ms=self._rf_len_ms)
                    cc_list = fit_metrics.cc
                    cc_norm_list = fit_metrics.cc_norm

                    for cc, cc_norm in zip(cc_list, cc_norm_list):
                        data_list.append({"pred_ms": pred_ms, "dim": f"{spatial_scale}_{max_dim}", "cc": cc.item(), "cc_norm": cc_norm.item()})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"ex: {str(e)}")
                    print(f"Could not build for pred_ms={pred_ms} spatial_scale={spatial_scale} max_dim={max_dim}.")

        return pd.DataFrame(data_list)
