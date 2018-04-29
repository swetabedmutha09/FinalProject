%% Motion-Based Multiple Object Tracking.
% This example is modified based on Matlabs Motion based Pedestrian
% tracking. The code structure is similar to matlabs original code
% structure.The current code is good for offline object tracking.
function multiObjectTracking_updated()

% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setupSystemObjects();

tracks = initializeTracks(); % Create an empty array of tracks.
trajectory = initializeTrajectory();
nextId = 1; % ID of the next track
count = 0;
% Detect moving objects, and track them across video frames.
srcFiles = dir('D:\Project_Final2017\dataset2012\dataset\baseline\highway\input1\*.jpg');  % the folder in which ur images exists
filename = ['D:\Project_Final2017\dataset2012\dataset\baseline\highway\input1\output' int2str(1) '.jpg'];
background_image = imread(filename);
background_grayimage = rgb2gray(background_image);
newbackground = background_grayimage;
%Find out image resolution
info = imfinfo(filename);
IMAGE_HEIGHT = info.Height;
IMAGE_WIDTH = info.Width;
n = length(srcFiles);
%Display of images is set to off. Instead images are saved in the local
%folder as its faster. The value in the set figure can be changed to 'on'
%in case if the individual want to see images as they are being processed.
f = figure;
set(f, 'visible', 'off');
for loop = 1 : 5
%     frame = readFrame();
    filename1 = strcat('D:\Project_Final2017\dataset2012\dataset\baseline\highway\input1\',srcFiles(loop).name);
    filename2 = strcat('D:\Project_Final2017\dataset2012\dataset\baseline\highway\input1\' ,srcFiles(loop+1).name);
    filename3 = strcat('D:\Project_Final2017\dataset2012\dataset\baseline\highway\input1\' ,srcFiles(loop+2).name);
    image1 = imread(filename1);
    image2 = imread(filename2);
    image3 = imread(filename3);
    grayimage1 = rgb2gray(image1);
    grayimage2 = rgb2gray(image2);
    grayimage3 = rgb2gray(image3);
    [centroids, bboxes] = detectObjects(grayimage1,grayimage2,grayimage3);
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();
    
    displayTrackingResults();
end


%% Create System Objects
% Create System objects used for reading the video frames, detecting
% foreground objects, and displaying results.

    function obj = setupSystemObjects()
        % Initialize Video I/O      
        % Create a video players, to display the video,
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);       
    end

%% Initialize Tracks

    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'centroid',{}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end

    function trajectory = initializeTrajectory()
        % create an empty array of tracks
        trajectory = struct(...
            'id', {}, ...
            'centroid',{});
    end

%% Detect Objects
% The |detectObjects| function returns the centroids and the bounding boxes
% of the detected objects. It also returns the binary mask, which has the 
% same size as the input frame. Pixels with a value of 1 correspond to the
% foreground, and pixels with a value of 0 correspond to the background.    

    function [centroids, bboxes] = detectObjects(grayimage1,grayimage2,grayimage3)
     %% 3 frame differencing algorithm   
     %Calculate the absolute difference of the two images.
        diff1 = imabsdiff(medfilt2(grayimage1),medfilt2(grayimage2));
        diff2 = imabsdiff(medfilt2(grayimage3),medfilt2(grayimage2));

        andImage = bitand(diff1,diff2);
    %     figure;imshow(mul3);

            %log edge detection
         BW = edge(andImage,'sobel');
    %      figure;imshow(BW);

        level = graythresh(andImage);
        D1 = im2bw(andImage,level);
    %      figure;imshow(D1)
        addimage = D1+BW;
    %       figure;imshow(addimage);
       %% Background subtraction algorithm
        diff3 = imabsdiff(medfilt2(newbackground),medfilt2(grayimage2));
        level = graythresh(diff3);
        Mkimage = im2bw(diff3,level);
    %      figure;imshow(Mkimage);
    % Background update algorithm. Background is updated every frame.
    %The value alpha is tunable. can be modified based on images or video used.
        alpha = 0.1;
        for ii = 1: IMAGE_HEIGHT
            for jj = 1:IMAGE_WIDTH
                if Mkimage(ii,jj) == 0
                    newbackground(ii,jj) = (alpha*grayimage2(ii,jj)) + ((1-alpha)*background_grayimage(ii,jj));
                else
                    newbackground(ii,jj) = background_grayimage(ii,jj);
                end
            end
        end
       background_grayimage = newbackground;
       %% Add the images generated by albove two algorithms and perform 
       % morphological operations.
       newimage = Mkimage + addimage;
    %    figure;imshow(newimage);
       BW2 = imopen(newimage, strel('square',3.0));
       windowSize = 8;
    %    figure;imshow(BW2);
       B3 = imclose(BW2, ones(windowSize));

       B4 = imfill(B3,'holes');
    %       figure;imshow(B4);
       bw = bwareaopen(B4,500);
    %         figure;imshow(bw);
       cc = bwconncomp(bw); 
       %region properties to get centroid and bounding box of the detected
       %objects in the images.
        stats = regionprops('table',cc,'Centroid','BoundingBox');
        centroids = stats.Centroid;
        bboxes = int32(stats.BoundingBox);

    end

%% Predict New Locations of Existing Tracks
% Use the Kalman filter to predict the centroid of each track in the
% current frame, and update its bounding box accordingly.

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            
            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);
            
            % Shift the bounding box so that its center is at 
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        
        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
    end

%% Update Assigned Tracks
% The |updateAssignedTracks| function updates each assigned track with the
% corresponding detection. It calls the |correct| method of
% |vision.KalmanFilter| to correct the location estimate. Next, it stores
% the new bounding box, and increases the age of the track and the total
% visible count by 1. Finally, the function sets the invisible count to 0. 

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);
            
            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;
            tracks(trackIdx).centroid = centroid;
            
            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
%             end
            
            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
%             if  tracks(ind).age < 3
            tracks(ind).age = tracks(ind).age + 1;
%             end
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

%% Delete Lost Tracks
% The |deleteLostTracks| function deletes tracks that have been invisible
% for too many consecutive frames. It also deletes recently created tracks
% that have been invisible for too many frames overall. 

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        
        invisibleForTooLong = 2;
        ageThreshold = 8;
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % Delete lost tracks.
        tracks = tracks(~lostInds);
%         trajectory = trajectory(~lostInds);
    end

%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

    function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        
        for i = 1:size(centroids, 1)
            
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            if(round(centroid(1,2))> 50)
            % Create a Kalman filter object.
                kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                    centroid, [200, 50], [100, 25], 100);

                % Create a new track.
                newTrack = struct(...
                    'id', nextId, ...
                    'bbox', bbox, ...
                    'centroid',centroid, ...
                    'kalmanFilter', kalmanFilter, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0);

                % Add it to the array of tracks.
                tracks(end + 1) = newTrack;

                % Increment the next id.
                nextId = nextId + 1;
            end
        end
    end

%% Display Tracking Results
% The |displayTrackingResults| function draws a bounding box and label ID 
% for each track on the video frame and the foreground mask. It then 
% displays the frame and the mask in their respective video players. 

    function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(image2);
%         mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        minVisibleCount = 1;
        if ~isempty(tracks)
              
            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than 
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            
            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);
                centroids = cat(1,reliableTracks.centroid);
                % Get ids.
                ids = int32([reliableTracks(:).id]);
                
                % Create labels for objects indicating the ones for 
                % which we display the predicted rather than the actual 
                % location.
                 labels = cellstr(int2str(ids'));
               
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                        bboxes, labels);
                %% Loop for counting the cars in the video: Uncomment the code if want to count
                % the number of cars in the video. Counting starts as soon
                % as they cross half of the image.
                 for cent = 1 : length(reliableTracks)
                     ybar = centroids(cent,2);
                     if (round(ybar) > (IMAGE_HEIGHT/2)&& round(ybar) < ((IMAGE_HEIGHT/2)+3))
                        count = count + 1;
                     end
                 end
                 %% Loop for saving the lables and centroid information for all the detected objects
                 %as soon as they enter the frame detection region.
                 for cent = 1 : length(reliableTracks)
                     id = ids(cent);
                     if(id == reliableTracks(cent).id)
                         trajectory(id).id = reliableTracks(cent).id;
                         centroid_temp = reliableTracks(cent).centroid;
                         ybar = centroid_temp(1,2);
                         if (round(ybar) > (50)&& round(ybar) < (300))
                            trajectory(id).centroid = [trajectory(id).centroid reliableTracks(cent).centroid];
                         end
                     end
                 end
            end
            text_str = ['Total Count: ' num2str(count)];
            frame = insertText(frame, [5 5], text_str, 'BoxOpacity', 1,'FontSize', 10);
        end
       
        % Display the frame.       
%         obj.videoPlayer.step(frame);
        %% loop for plotting the trajectory and speed plot
        if ~isempty(tracks)
            imagesc(frame);hold on;
            CM = jet(length(reliableTracks));
            for traj = 1: length(reliableTracks)
                id = ids(traj);
                centroid_traj = trajectory(id).centroid;
                %%loop for plotting the trajectory:Uncomment the below code if want to plot trajectory graph
                for traj1 = 1: 2: length(centroid_traj)
                    xbar = centroid_traj(1,traj1);
                    ybar = centroid_traj(1,traj1+1);
                    plot(xbar, ybar, 'color', CM(traj,:),'marker','o');
                end
                %% loop for plotting the speed of the car: Uncomment the below code if want to plot speed graph.
                % All the calculations are done based on assumption the fps
                % for the video is 25.So time to process one frame is
                % 0.4secs.
                %followed the calculations based on :
                %http://www.academia.edu/1002902/Vehicle_speed_detection_in_video_image_sequences_using_CVS_method
%                 distcount = 0;
%                 timecount = 1;
%                 time(1) = 6;
%                 if length(centroid_traj) == 60
%                     for traj1 = 9: 2: length(centroid_traj)-2
%                         distcount = distcount + 1;
%                         xbar = centroid_traj(1,traj1);
%                         ybar = centroid_traj(1,traj1+1);
%                         xbar1 = centroid_traj(1,traj1+2);
%                         ybar1 = centroid_traj(1,traj1+3);
%                         dist = sqrt(((xbar1 - xbar)^2)+((ybar1 - ybar)^2));
                          %% calibration value 0.07 is a pixel distance.
                          % 13 meters of distance is assumed for detection region.In image that region is 190 pixles.
                          % therefore to convert actual distance to 2d
                          % image distance 13/190 i.e. approx 0.07
%                         dist = (dist*0.07)/0.04;
%                         dist = (dist * 18)/5;
%                         distance(distcount)= dist; 
%                         if(distcount>1)
% %                             time(distcount) = time(distcount-1)+ timecount;
%                             time(distcount) = time(distcount-1)+ timecount;
%                         end
%                     end
%                     plot(time,distance,'r');
%                     title('2-D Vehicle Speed Plot')
%                     xlabel('frames')
%                     ylabel('Units')
%                     saveas(f,['D:\Project_Final2017\dataset2012\dataset\baseline\highway\speedplot\Output' num2str(id) '.jpg']);
%                 end
            end
            % save as function for saving trajectory plot: Must be
            % uncommented when plotting the trajectory graph above
             saveas(f,['D:\Project_Final2017\dataset2012\dataset\baseline\highway\trajplot\Output' num2str(loop) '.jpg']);
              hold off;
        end
    end

displayEndOfDemoMessage(mfilename)
end



