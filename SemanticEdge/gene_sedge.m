% --------------------------------------------------------
% Intro:
% This script is used to:
% 1. Generate instance-sensitive multi-label semantic edges on the Cityscapes dataset
% 2. Create filelists for the generated data and labels
% --------------------------------------------------------

function gene_sedge()
clc; clear; close all;

%% Add library paths
path = genpath('../../../lib/matlab');
addpath(path);
%% 

%% Setup Directories and Suffixes
dataRoot = 'path'; %'../data_orig';
genDataRoot = 'path_edge'; % debug
suffixImage = '_label_edge.png';

%% Setup Parameters
numCls = 6;
radius = 2;
flagPngFile = true; % Output .png edge label files

%% Setup Parallel Pool
numWorker = 3; % Number of matlab workers for parallel computing
delete(gcp('nocreate'));
parpool('local', numWorker);

%% Generate Output Directory
if(exist(genDataRoot, 'file')==0)
    mkdir(genDataRoot);
end

%% Preprocess Training Data and Labels
setList = {'train', 'test'};
list = [5, 4, 3, 2, 1, 0];
for idxSet = 1:length(setList)
    setName = setList{idxSet};
    if(flagPngFile)
        fidList = fopen([genDataRoot '/' setName '_sedge_labels.txt'], 'w');
    end
    dataList = cell(1, 1);
    countFile = 0;
    if(exist([genDataRoot '/' setName '_sedge_labels'], 'file')==0)
        mkdir([genDataRoot '/' setName '_sedge_labels']);
    end
    fileList = dir([dataRoot '/' setName '_sedge_labels_resize_new/*.png']);

    % Generate and write data
    parfor_progress(length(fileList));
    parfor idxFile = 1:length(fileList)
        assert(strcmp(fileList(idxFile).name(end-length(suffixImage)+1:end), suffixImage), 'suffixImage mismatch!')
        fileName = fileList(idxFile).name(1:end-length(suffixImage));

        if(1)
            % Transform label id map to train id map and write
            % labelIdMap = imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds]);
            instIdMap = imread([dataRoot '/' setName '_sedge_labels_resize/' fileName suffixImage]);
            % trainIdMap = labelid2trainid(labelIdMap);
            % imwrite(trainIdMap, [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixTrainIds], 'png');
            % Transform color map to edge map and write
            % edgeMapBin = seg2edge(instIdMap, radius, []', 'regular'); % Avoid generating edges on "rectification border" (labelId==2) and "out of roi" (labelId==3)
            [height, width] = size(instIdMap);
            % labelEdge = cell(numCls, 1);
            % labelEdge2 = zeros(height, width, 'uint32');

            % Generate .png edge label
            labelEdge_b = zeros(height, width, 'uint8');
            labelEdge_g = zeros(height, width, 'uint8');
            labelEdge_r = zeros(height, width, 'uint8');
            labelEdge_png = zeros(height, width, 3, 'uint8');
            for idxCls = 1:numCls
                idxSeg = instIdMap == list(idxCls);
                if(sum(idxSeg(:))~=0 || list(idxCls)==0)
                    segMap = zeros(size(instIdMap));
                    segMap(idxSeg) = instIdMap(idxSeg);
                    % idxEdge = seg2edge_fast(segMap, edgeMapBin, radius, [], 'regular');
                    idxEdge = seg2edge(segMap, radius, []', 'regular');
                    % idxEdge = edgeMapBin;
                    idxEdge = idxEdge & instIdMap
               
                    % labelEdge{idxCls, 1} = sparse(idxEdge);
                    % labelEdge2(idxEdge) = labelEdge2(idxEdge) + 2^(idxCls-1);
                    if list(idxCls)==1
                        labelEdge_r(idxEdge) = 128;
                        labelEdge_g(idxEdge) = 0;
                        labelEdge_b(idxEdge) = 0;
                    elseif list(idxCls)==2
                        labelEdge_r(idxEdge) = 0;
                        labelEdge_g(idxEdge) = 0;
                        labelEdge_b(idxEdge) = 128;
                    elseif list(idxCls)==3
                        labelEdge_r(idxEdge) = 128;
                        labelEdge_g(idxEdge) = 0;
                        labelEdge_b(idxEdge) = 128;
                    elseif list(idxCls)==4
                        labelEdge_r(idxEdge) = 0;
                        labelEdge_g(idxEdge) = 128;
                        labelEdge_b(idxEdge) = 128;
                    elseif list(idxCls)==5
                        labelEdge_r(idxEdge) = 64;
                        labelEdge_g(idxEdge) = 0;
                        labelEdge_b(idxEdge) = 0;
                    end
                else
                    % labelEdge{idxCls, 1} = sparse(false(height, width));
                    continue;
                end
            end
            labelEdge_png = cat(3, labelEdge_r, labelEdge_g, labelEdge_b);

            if(flagPngFile)
                imwrite(labelEdge_png, [genDataRoot '/' setName '_sedge_labels_new/' fileName suffixImage], 'png');
            end
           % savelabeledge([genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixEdge], labelEdge); % parfor does not support directly using save.
        end
        parfor_progress();
    end
    parfor_progress(0);

    % Create file lists
%     for idxFile = 1:length(fileList)
%         countFile = countFile + 1;
%         fileName = fileList(idxFile).name(1:end-length(suffixImage));
%         if(ismember(setName, {'train', 'val'}))
%             if(flagPngFile)
%                 fprintf(fidList, ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage ' /gtFine/' setName '/' cityName '/' fileName suffixEdge_png '\n']);
%             end
%             dataList{countFile, 1} = ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage];
%             dataList{countFile, 2} = ['/gtFine/' setName '/' cityName '/' fileName suffixEdge];
%         else
%             if(flagPngFile)
%                 fprintf(fidList, ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage '\n']);
%             end
%             dataList{countFile, 1} = ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage];
%         end
%     end
%     if(flagPngFile)
%         fclose(fidList); %#ok<*UNRCH>
%     end
%     save([genDataRoot '/' setName '.mat'], 'dataList');
end
