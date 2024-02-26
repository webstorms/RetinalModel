import argparse

import torch
import torch.nn.functional as F
import scipy.io


class BuildTensors:

    CELL_NAMES = ["off parasol1",
                  "off parasol2",
                  # "off parasol3", <- some stimuli don't have 5 repeats
                  # "off parasol4", <- some stimuli don't have 5 repeats
                  # "off parasol5", <- some stimuli don't have 5 repeats
                  "off parasol6",
                  "off parasol7",
                  "off parasol8",
                  "off parasol9",
                  "off parasol10",
                  "on parasol1",
                  "on parasol2",
                  "on parasol3",
                  "on parasol4",
                  "on parasol5",
                  "on parasol6",
                  "on parasol7",
                  "on parasol8",
                  "on parasol9"
                  ]

    def __init__(self, root):
        self._root = root

        # Build x
        x = self._build_x()
        torch.save(x, f"{root}/data/neural/macaque_image/tensors/x.pt")

        # Build ys
        for cell_name in BuildTensors.CELL_NAMES:
            y = self._build_y(cell_name)
            assert list(y.shape) == [48, 5, 7500]
            torch.save(y, f"{root}/data/neural/macaque_image/tensors/{cell_name}.pt")

    def _build_x(self):
        clip_list = []

        # Any file will do as stimuli is the same
        mat = scipy.io.loadmat(f"{self._root}/data/neural/macaque_image/image_mats/on parasol4.mat")

        for i in range(48):
            img_mat = mat["databaseImport"][:, 2][i]
            img = torch.from_numpy(img_mat)[159:-160, 109:-110].float().permute(1, 0)
            clip = img.unsqueeze(0).repeat(15, 1, 1)
            clip = F.pad(clip, (0, 0, 0, 0, 15, 15), value=41.6646)
            clip_list.append(clip)

        return torch.stack(clip_list)

    def _build_y(self, cell_name):
        print(f"cell_name={cell_name}")
        spike_list = []

        mat = scipy.io.loadmat(f"{self._root}/data/neural/macaque_image/image_mats/{cell_name}.mat")

        for i in range(48):
            spike_mat = mat["databaseImport"][:, 1][i]
            spikes = torch.from_numpy(spike_mat)
            spike_list.append(spikes)

        return torch.stack(spike_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()
    BuildTensors(args.root)
