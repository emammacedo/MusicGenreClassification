function percentage = class_percentage (labels)

%Calculates the presence of each class in the dataset
%
%   Inputs:
%   labels -> original patterns labels
%
%   Outputs:
%   percentage -> percentage of each class in the data

total = length(labels);
classes = string(unique(labels));

percentage = cell(length(classes),2);

for i=1:length(classes)
    no_cases = length(find(labels == classes(i)));
    percentage{i,1} = classes(i);
    percentage{i,2} = no_cases/total*100;
end
end