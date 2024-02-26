import ReduceNaturalScenes.*;

export_path = '../RetinalModel/data/neural/macaque_movie/movie_mats';  % Change this to your path

centerSigma = 70;
surroundSigma = 170;
obj = ReduceNaturalScenes.demoUtils.loadSettings(centerSigma,surroundSigma);
obj.naturalDisks = 1;
obj.diskRegions = [0, 41];

obj.imageNo = 100;
projection = ReduceNaturalScenes.generateProjection(obj, 0).projection;
save(fullfile(export_path, '00034.mat'), "projection");

obj.imageNo = 81;
projection = ReduceNaturalScenes.generateProjection(obj, 0).projection;
save(fullfile(export_path, '00152.mat'), "projection");

obj.imageNo = 79;
projection = ReduceNaturalScenes.generateProjection(obj, 0).projection;
save(fullfile(export_path, '00161.mat'), "projection");

obj.imageNo = 73;
projection = ReduceNaturalScenes.generateProjection(obj, 0).projection;
save(fullfile(export_path, '00190.mat'), "projection");

obj.imageNo = 71;
projection = ReduceNaturalScenes.generateProjection(obj, 0).projection;
save(fullfile(export_path, '00195.mat'), "projection");

obj.imageNo = 12;
projection = ReduceNaturalScenes.generateProjection(obj, 0).projection;
save(fullfile(export_path, '01366.mat'), "projection");

obj.imageNo = 5;
projection = ReduceNaturalScenes.generateProjection(obj, 0).projection;
save(fullfile(export_path, '04103.mat'), "projection");