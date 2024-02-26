% Modified verion of the original compareImageFlashses.m script by J. Freedland & F. Rieke.

clear
clc
% VARIABLES TO BE MODIFIED:
cellType = 'off parasol';  % Run for 'on parasol' and 'off parasol'
folderName = "/Users/home/desktop/image_mats";  % Change to your home path

mkdir(folderName);
load(strcat('+Figures/+Figure2/+Data/+CenterImageFlashes/',cellType,'_imageFlashes.mat'))

for cellNumber = 1:size(dataset,1)
    display(cellNumber);
    
    onsetSpikes = dataset{cellNumber,5}{1};
    
    % Key parameters for experiment
    experimentSettings = cell(size(dataset{cellNumber,4}{1},1),3);
    for i = 1:size(dataset{cellNumber,4}{1},1)
        experimentSettings{i,1} = mat2str(dataset{cellNumber,4}{1}{i,1}.imageNo);
        experimentSettings{i,2} = mat2str(dataset{cellNumber,4}{1}{i,1}.frameNo);   
        experimentSettings{i,3} = '0';
    end 
    ii = Figures.Utils.uniqueExperimentIndex(experimentSettings);
    
    % Average across trials
    modelingImport = cell(max(ii),2);
    experimentSettings = zeros(max(ii),3);
    for i = 1:max(ii)
        trials = find(i == ii);

        % Import full experiment parameters
        modelingImport{i,1} = dataset{cellNumber,4}{1}{trials(1),1}; 
        
        % Average spikes
        modelingImport{i,2} = onsetSpikes(trials, :);
        
        % For distinguishing between flashed natural images and flashed reduced images
        experimentSettings(i,1) = modelingImport{i,1}.imageNo;
        experimentSettings(i,2) = modelingImport{i,1}.frameNo;
        experimentSettings(i,3) = modelingImport{i,1}.slices;
    end
    
    databases = NaN;
    
    % Replace metadata with actual flashed image
    display(length(databases));
    for a = 1:length(databases)
        databaseImport = modelingImport;
        

        for b = 1:size(databaseImport,1)
            
            clear obj
            obj = databaseImport{b,1};
            obj.rfSigmaCenter = 70;
            obj.rfSigmaSurround = 170;
            obj.diskRegions = [1, 40];
            display(obj);
            [path,img] = ReduceNaturalScenes.utils.pathDOVES(obj.imageNo, obj.observerNo);
            
            % Normalize image to monitor
            img = (img./max(max(img))) .* 255;
            obj.imageMatrix = uint8(img);
            
            % Mean light intensity for retinal adaptation.
            obj.backgroundIntensity = 0.168 * 255; % Parameters for experiment
            
            % Eye movement patterns from DOVES database.
            obj.xTraj = path.x(obj.frameNo);
            obj.yTraj = path.y(obj.frameNo);

            % Calculate individual neuron's receptive field (RF).
            [RFFilter,obj.rfSizing] = ReduceNaturalScenes.rfUtils.calculateFilter(obj);
            
            % Convolve filter with trajectory
            [weightedTraj, stimulus.raw] = ReduceNaturalScenes.utils.weightedTrajectory(obj, img, RFFilter);
            
            % Calculate disks
            [stimulus.projection, stimulus.values, stimulus.masks] = ReduceNaturalScenes.utils.linearEquivalency(obj, weightedTraj, RFFilter, stimulus.raw);
            databaseImport{b,3} = stimulus.projection';
            
        end
        
        save(fullfile(folderName, [cellType, num2str(cellNumber), '.mat']), 'databaseImport');
        
    end
end
