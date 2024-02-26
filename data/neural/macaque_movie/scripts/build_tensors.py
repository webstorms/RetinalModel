import glob
import argparse

import torch
import torch.nn.functional as F
import scipy.io
import pandas as pd
import numpy as np


class BuildTensors:

    def __init__(self, root):
        self._root = root
        self._query_df, self._spikes_dict = self._build_query_df_and_spikes_dict()

        # Build x
        x = self._build_x()
        torch.save(x, f"{root}/data/neural/macaque_movie/tensors/x.pt")

        # Build ys
        for cell_name in self._query_df["cell_names"].unique():
            y = self._build_y(cell_name)

            if y.shape[0] == 7:  # Has all seven stimuli
                torch.save(y, f"{root}/data/neural/macaque_movie/tensors/{cell_name}.pt")

    def _build_x(self):
        stim_names = self._query_df["stim_names"].unique()
        stim_names = np.sort(stim_names)

        x_list = []

        for stim_name in stim_names:
            mat = scipy.io.loadmat(f"{self._root}/data/neural/macaque_movie/movie_mats/{stim_name}")

            movie_tensor = torch.from_numpy(mat["projection"][:, :, 0])
            movie_tensor = movie_tensor.permute(2, 0, 1)

            n_frame_diff = 330 - movie_tensor.shape[0]
            frame_padding = torch.stack([movie_tensor[-1] for _ in range(n_frame_diff)])
            movie_tensor = torch.cat([movie_tensor, frame_padding])
            movie_tensor = F.pad(movie_tensor, (0, 0, 0, 0, 15, 15), value=41.6646)
            x_list.append(movie_tensor)

        return torch.stack(x_list)

    def _build_y(self, cell_name):
        query = self._query_df["cell_names"] == cell_name
        df = self._query_df[query]
        df = df.sort_values("stim_names")
        cell_name = df["cell_names"].unique()[0]

        spike_list = []
        min_repeats = 1000000

        for mat_name in df["mat_names"]:
            spikes = self._spikes_dict[cell_name][mat_name]
            spikes = torch.from_numpy(spikes).float()
            spike_list.append(spikes)
            if spikes.shape[0] < min_repeats:
                min_repeats = spikes.shape[0]

        # Trim to minimum repeats so we can build tensors...
        spike_list = [spike_tensor[:min_repeats] for spike_tensor in spike_list]

        return torch.stack(spike_list)

    def _build_query_df_and_spikes_dict(self):
        mat_names = []
        stim_names = []
        cell_names = []
        spikes_dict = {}

        for path in glob.glob(f"{self._root}/data/neural/macaque_movie/processed_mats/*"):
            mat_name = path.split("/")[-1]
            stim_name = path.split("/")[-1].split("_")[1][3:]
            m = scipy.io.loadmat(path)
            cell_idxs = m["data"][0][0][4][0][0]
            dates = m["data"][0][0][4][0][1]
            video_settings = m["data"][0][0][4][0][3]
            spikes = m["data"][0][0][4][0][4]

            for i in range(len(cell_idxs)):
                is_natural = len(video_settings[i][0][0][0][17]) > 0
                date = dates[i][0][0]
                cell_idx = cell_idxs[i][0][0]

                if is_natural:
                    unique_idx = f"{cell_idx}_{date}"
                    mat_names.append(mat_name)
                    stim_names.append(stim_name)
                    cell_names.append(unique_idx)

                    if spikes_dict.get(unique_idx) is None:
                        spikes_dict[unique_idx] = {}

                    spikes_dict[unique_idx][mat_name] = spikes[i][0]

        query_df = pd.DataFrame({"mat_names": mat_names, "stim_names": stim_names, "cell_names": cell_names})

        return query_df, spikes_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()
    BuildTensors(args.root)
