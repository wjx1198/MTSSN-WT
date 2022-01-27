%% Setup Directories and Suffixes
dataRoot = 'path'; %'../data_orig';
genDataRoot = 'path_edge'; % debug
suffixImage = '_label_edge.png';

setList = {'train', 'test'};
for idxSet = 1:length(setList)
    setName = setList{idxSet};
    fileList = dir(['F:/DFF-master/edge_proc/' setName '_sedge_labels_new/*.png']);
    for idxFile = 1:length(fileList)
        assert(strcmp(fileList(idxFile).name(end-length(suffixImage)+1:end), suffixImage), 'suffixImage mismatch!')
        fileName = fileList(idxFile).name(1:end-length(suffixImage));
        i=imread(['F:/DFF-master/edge_proc/' setName '_sedge_labels_new/' fileName suffixImage]);
        [m,n,q]=size(i);
        for x=1:m
            for y=1:n
                if i(x,y,1)==0 && i(x,y,2)==0 && i(x,y,3)==0
                    i(x,y,1)=255;
                    i(x,y,2)=255;
                    i(x,y,3)=255;
                end
            end
        end
        imwrite(i, ['F:/DFF-master/edge_proc/edge/' fileName suffixImage], 'png');
    end
end
