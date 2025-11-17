path = '../RetinalModel/data/neural/macaque_movie/Systematic-reduction_dataset/+Figures/+Figure1/+Data/+ReducedMovies'; % Input path
processed_path = '../RetinalModel/data/neural/macaque_movie/processed_mats'; % Processed data path

files = dir(fullfile(path, '*.mat'));

for i = 1:length(files)
    file_name = files(i).name;
    file_path = fullfile(path, file_name);
    
    % Load the dataset
    dataset = load(file_path);
    data = struct(dataset);
    
    % Save processed data
    save(fullfile(processed_path, file_name), 'data.dataset');
end