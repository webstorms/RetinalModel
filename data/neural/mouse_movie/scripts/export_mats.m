% Ensure the stimulus_reconstruction code and downsample.m are in the path
% downsample.m found under marmoset_movie/scripts
data_path = '../RetinalModel/data/neural/mouse_movie/10.12751_g-node.ejk8kx'; % Input path
export_path = '../RetinalModel/data/neural/mouse_movie/mats';  % Processed data path


file_paths = [fullfile(data_path, "20210115_252MEA_mouse_left_half_ventral", "fixationmovie_data.mat"),
    fullfile(data_path, "20210115_252MEA_mouse_right_half_ventral", "fixationmovie_data.mat"),
    fullfile(data_path, "20210203_252MEA_mouse_left_half_dorsal", "fixationmovie_data.mat"),
    fullfile(data_path, "20210203_252MEA_mouse_left_half_ventral", "fixationmovie_data.mat"),
    fullfile(data_path, "20210803_252MEA_mouse_left_half_dorsal", "fixationmovie_data.mat"),
    fullfile(data_path, "20210813_252MEA_mouse_left_half_dorsal", "fixationmovie_data.mat"),
    fullfile(data_path, "20210813_252MEA_mouse_left_half_ventral", "fixationmovie_data.mat"),
    fullfile(data_path, "20210916_252MEA_mouse_right_half_ventral", "fixationmovie_data.mat")
    ];
targetHeight = 150;
targetWidth = 200;

for index = 1:numel(file_paths)
    file_path = file_paths{index};
    load(file_path);
    y_name = strcat('y', num2str(index), ".mat");

    if index == 1
        x_name = strcat('x', num2str(index), ".mat");
        blockstimulus = returnFixMovie([600, 800], frozenImages, frozenfixations);
        downsampledMovie = downsample(blockstimulus, targetHeight, targetWidth);
        save(fullfile(export_path, x_name), "downsampledMovie");
    end
    save(fullfile(export_path, y_name), "frozenbin");
end