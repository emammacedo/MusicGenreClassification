function data_dev = feat_sel_corr (data_dev ,H_values, lim)

%Computes the correlation matrix and, for each pair of high correlated 
%features, scraps the one with lower ranking (less discriminative)
%
%   Inputs:
%   data -> data structure
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .feat_name [1 x dim] -> names of the features
%       .idx_og [1 x dim] -> indexes of the features in the initial dataset
%   num_feat -> number of features to select
%   H_val [1 x num_feat] -> sorted ranking of the features selected
%   lim -> higher correlation value accepted
%
%   Outputs:
%   data_dev_kw -> data structure with selected features
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .feat_name [1 x num_feat] -> names of the features
%       .idx_og [1 x num_feat] -> indexes of the features in the initial dataset
%       .st [2 x dim]-> meand and variance of each feature~

corr_matrix = corrcoef(data_dev.X');
%h = heatmap(corr_matrix,'MissingDataColor','w');
%corrplot(data_dev.X');

idx_remove = [];
for i=2:size(corr_matrix,2)
    for j=1:(i-1)
        if corr_matrix(j,i) >= lim
%             if H_values(j) < H_values(i)
%                 idx_matrix = [idx_matrix j];
%             else
%                 idx_matrix = [idx_matrix i];
%             end
            idx_remove = [idx_remove i];
        end
    end
end

idx_remove = sort(unique (idx_remove), 'descend');

for i=1:length(idx_remove)
    data_dev.X(idx_remove(i),:)=[];
    data_dev.feat_name(idx_remove(i))=[];
    data_dev.idx_og(idx_remove(i))=[];
    data_dev.st(:,idx_remove(i)) = [];
end
data_dev.dim = size(data_dev.X, 1);
end
