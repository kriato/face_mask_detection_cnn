%% TRAINING SET
data = load('train_label.mat').label_train;
filename = 'train_label.csv';

fileID = fopen(filename,'w');

for i = 1:size(data, 2)
    for j = 1:size(data(i).label,1)
        fprintf(fileID, '%s;%s', data(i).orgImgName, data(i).imgName);
            for k = 1:size(data(i).label,2)
                fprintf(fileID, ';%d', data(i).label(j,k));
            end
            fprintf(fileID, '\n');
    end
end

fclose(fileID);

%% TESTING SET
data = load('test_label.mat').LabelTest; 
filename = 'test_label.csv';

fileID = fopen(filename,'w');

for i = 1:size(data, 2)
    for j = 1:size(data(i).label,1)
        fprintf(fileID, '%s', data(i).name);
            for k = 1:size(data(i).label,2)
                fprintf(fileID, ';%d', data(i).label(j,k));
            end
            fprintf(fileID, '\n');
    end
end

fclose(fileID);