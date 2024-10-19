function [features_values, data_label, data_label_numeric, features_names] = read_data()

%Reads the original dataset, extracts information (features values, patterns'
%labels and features names) and transforms the labels into numeric values.
%
%   Output:
%   features_values -> matrix with features values
%   data_label -> matrix with the original labels of the patterns
%   data_label_numeric -> matrix with the numeric labels of the patterns
%   features_names -> cell with the features names
%

data_table = readtable('/Users/Asus/Desktop/RP/Projeto/Data/dados.csv'); 
features_names = data_table.Properties.VariableNames(2:198); 
features_values = data_table {:,2:198}; 
data_label = data_table{:,199}; %patterns classification
data_label_numeric = []; %patterns numeric classification

classes = string(unique(data_label));
for i=1:length(data_label)
    for j=1:length(classes)
        if data_label(i) == classes(j)
            data_label_numeric = [data_label_numeric; j];
        end
    end
end

%NOTE:
%the first column of the data_table was not included in features_values, 
%since it corresponds to the the name of each audio file

end