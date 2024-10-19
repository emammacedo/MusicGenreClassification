function [data_dev_kw, H_val] = feat_sel_kw (data_dev, num_feat)

%Performs the Kruskal-Wallis Test (that ranks the features according 
%to their discriminative power) and selects the best ones
%
%   Inputs:
%   data -> data structure
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .feat_name [1 x dim] -> names of the features
%       .idx_og [1 x dim] -> indexes of the features in the initial dataset
%   num_feat -> number of features to select
%
%   Outputs:
%   data_dev_kw -> data structure with selected features
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .feat_name [1 x num_feat] -> names of the features
%       .idx_og [1 x num_feat] -> indexes of the features in the initial dataset
%       .st [2 x dim]-> meand and variance of each feature
%   H_val [1 x num_feat] -> sorted ranking of the features selected

ranking=cell(data_dev.dim,2);
for i=1:data_dev.dim
    [p,atab,stats]=kruskalwallis(data_dev.X(i,:),data_dev.y,'off');
    ranking{i,1}=data_dev.feat_name{i};
    ranking{i,2}=atab{2,5};
end

[H_val,Ind] = sort([ranking{:,2}],2,'descend');

% stotal=[sprintf('K-W Feature ranking:\n')];%Get a string with the rankings
% for i=1:data_dev.dim
%     stotal=[stotal,sprintf('%s-->%.2f\n',ranking{Ind(i),1},ranking{Ind(i),2})];
% end
% disp(stotal)%Diplay a table with the K-W feature ranking

data_dev_kw = data_dev;
matrix_aux = [];
names_aux = cell(1,num_feat); %features to select

for i=1:num_feat
    matrix_aux(i,:) = data_dev.X(Ind(i),:); 
    names_aux{1,i} = data_dev.feat_name{Ind(i)}; 
end

data_dev_kw.X=matrix_aux;
data_dev_kw.feat_name=names_aux;
data_dev_kw.dim = size(data_dev.X,1);
%data_dev_kw.idx_og = Ind(1:num_feat); %%%
data_dev_kw.idx_og = data_dev.idx_og(Ind(1:num_feat)); 
data_dev_kw.st = data_dev.st(:,Ind(1:num_feat)); 

H_val = H_val(1:num_feat);

end