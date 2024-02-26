function downsampledMovie = downsample(blockstimulus, targetHeight, targetWidth)

[height, width, frames] = size(blockstimulus);

% Initialize a new tensor to store the downsampled movie
downsampledMovie = zeros(targetHeight, targetWidth, frames, 'single');

% Perform bilinear downsampling frame by frame
for f = 1:frames
    % Perform downsampling using the 'imresize' function
    downsampledFrame = imresize(blockstimulus(:,:,f), [targetHeight, targetWidth], 'bilinear');

    % Store the downsampled frame in the new tensor
    downsampledMovie(:,:,f) = downsampledFrame;
end

end