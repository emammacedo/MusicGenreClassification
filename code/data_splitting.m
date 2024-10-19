function  [data_dev, data_test] = data_splitting (features_values, data_label_numeric, features_names, dev_percent)

%Splits the data into development and testing sets randomly and according
%with a certain percentage.
%
%   Inputs:
%   features_values -> matrix with features values
%   data_label_numeric -> matrix with the numeric labels of the patterns
%   features_names -> cell with the features names
%   dev_percent -> percentage of development test
%
%   Outputs:
%   data_dev -> data structure
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .dim -> no. of features
%       .num_data -> no. of patterns
%       .feat_name [1 x dim] -> names of the features
%       .idx_og [1 x dim] -> indexes of the features in the initial dataset
%   data_test -> data structure
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .dim -> no. of features
%       .num_data -> no. of patterns

idx_rand = randperm(length(features_values)); %random indexes
idx_sep=floor(dev_percent*length(features_values)); 
%index that ditactes where the separation between develop. and test occurs

%data for development
data_dev.X = (features_values(idx_rand(1:idx_sep),:))';
data_dev.y = data_label_numeric(idx_rand(1:idx_sep),:);
data_dev.dim = size(data_dev.X,1);
data_dev.num_data = size(data_dev.X,2);
data_dev.feat_name = features_names;
data_dev.idx_og = 1:data_dev.dim; 

%data for testing
data_test.X = (features_values(idx_rand(idx_sep+1:999),:))'; 
data_test.y = data_label_numeric(idx_rand(idx_sep+1:999),:); 
data_test.dim = size(data_test.X,1);
data_test.num_data = size(data_test.X,2); 

end