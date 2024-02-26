import retina.neural.dataset.movie.macaque as macaque_movie
import retina.neural.dataset.movie.marmoset as marmoset_movie
import retina.neural.dataset.movie.mouse as mouse_movie
import retina.neural.dataset.image.salamander as salamander_image
import retina.neural.dataset.image.macaque as macaque_image


def load(root, train, dataset_name, spatial_scale, max_dim, luminance, cell_idx):
    if dataset_name == "MacaqueMovie":
        return macaque_movie.MacaqueMovie(root, train, spatial_scale, max_dim, luminance=luminance, cell_idx=cell_idx)
    elif dataset_name == "MarmosetMovie":
        return marmoset_movie.MarmosetMovie(root, train, spatial_scale, max_dim, luminance=luminance, cell_idx=cell_idx)
    elif dataset_name == "MouseMovie":
        return mouse_movie.MouseMovie(root, train, spatial_scale, max_dim, luminance=luminance, cell_idx=cell_idx)
    elif dataset_name == "SalamanderImage":
        return salamander_image.SalamanderImage(root, train, spatial_scale, max_dim, luminance=luminance, cell_idx=cell_idx)
    elif dataset_name == "MacaqueImage":
        return macaque_image.MacaqueImage(root, train, spatial_scale, max_dim, luminance=luminance, cell_idx=cell_idx)

    raise NotImplementedError("Could not load requested dataset.")