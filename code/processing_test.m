function data = processing_test(data_test, data_dev)

%Applys the features transformations done to the development data to the
%test data, including the feature selection, the standardization and the 
%pca analysis
%
%   Inputs:
%   data_dev -> data structure
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .feat_name [1 x dim] -> names of the features
%       .idx_og [1 x dim] -> indexes of the features in the initial dataset
%       .st [2 x dim]-> meand and variance of each feature
%       .pca_model [struct]-> pca model (to be applied to test data)
%   data_test -> data structure
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%
%   Outputs:
%   data -> data structure after processing
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels

data = data_test;

%feature selection
matrix_aux = [];
for i=1:length(data_dev.idx_og);
    matrix_aux(i,:) = data.X(data_dev.idx_og(i),:); 
end
data.X=matrix_aux;
data.dim = size(data.X,1);

%standardization with mean and std from development data
for i=1:data.dim
    for j=1:data.num_data
        data.X(i,j) = (data.X(i,j)-data_dev.st(1,i))/data_dev.st(2,i);
    end
end

if isfield(data_dev, 'pca_model') == 1 %pca
    data.X = linproj(data.X, data_dev.pca_model);
elseif isfield(data_dev, 'lda_model') == 1 %lda
    
    if isfield(data_dev.lda_model, 'class') == 1 %binary classif
        class = data_dev.lda_model.class;
        idx_all = find(data.y ~= class);
        idx_one = find(data.y == class);
        data.y(idx_one) = 1;
        data.y(idx_all) = 2;
    end
    
    data.X = linproj(data.X, data_dev.lda_model);
end
data.dim = size(data.X,1);

end