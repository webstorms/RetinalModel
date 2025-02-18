import torch
import pandas as pd

from retina import train
import retina.dataset as retina_dataset


class FeedforwardVsRecurrentContribution:

    def __init__(self, root, model_name="0.0031622776601683794_*_0.01_0.6_128_8", dataset_path="/home/datasets/natural"):
        self.model = train.Trainer.load_model(f"{root}/results", model_name)
        dataset = retina_dataset.PatchNaturalDataset(root=dataset_path, train=False, temp_len=600, kernel=20, flip=True, n_frame_ext=2)
        self.clips = torch.stack([dataset[i][0] for i in range(10)])
        self.mean_fwd_current_per_clip, self.mean_rec_current_per_clip = self.get_currents()
        self.mean_fwd_current_per_clip = self.mean_fwd_current_per_clip.mean((0, 1, 3, 4, 5))
        self.mean_rec_current_per_clip = self.mean_rec_current_per_clip.mean((0, 1, 3))

    def get_currents(self):
        fwd_current_list = []
        rec_current_list = []

        for i in range(self.clips.shape[0]):
            with torch.no_grad():
                abs_graded_current, abs_rec_current = self.model(self.clips[i:i+1].cuda(), mode="val_curr", stride=4)
                fwd_current_list.append(abs_graded_current)
                rec_current_list.append(abs_rec_current)

        fwd_current = torch.stack(fwd_current_list).cpu()
        rec_current = torch.stack(rec_current_list).cpu()

        return fwd_current, rec_current

    @staticmethod
    def get_df(root, prediction_offset=128):
        current_analysis = FeedforwardVsRecurrentContribution(root, model_name=f"0.0031622776601683794_*_0.01_0.6_{prediction_offset}_8")
        current_analysis_df = pd.DataFrame({"Feedforward": current_analysis.mean_fwd_current_per_clip, "Recurrent": current_analysis.mean_rec_current_per_clip})

        return current_analysis_df
