function data_lda = feat_red_lda(data_dev, dim, class)

%LDA takes into account the labels, so we need to specifiy if we are doing
%binary or multi-class classificiation

%class --> class we want to classify against all others

if nargin == 3 %for binary classification, we specify wich class we want
    idx_all = find(data_dev.y ~= class);
    idx_one = find(data_dev.y == class);
    data_dev.y(idx_one) = 1;
    data_dev.y(idx_all) = 2;
    dim = 1; %caso haja algum erro
end

model_reduction = lda(data_dev, dim); 

data_lda=data_dev;
data_lda.X=linproj(data_dev.X,model_reduction);
data_lda.dim=dim;

if exist('class','var') %doesn´t exist for multi class
    model_reduction.class = class;
end
data_lda.lda_model = model_reduction;

% %plot
% if dim<=3
%     figure()
%     ppatterns(data_lda)
%     xlabel('LDA1')
%     ylabel('LDA2')
%     if dim == 3
%         zlabel('LDA3')
%     end
%     title(['LDA - ' num2str(dim) ' dimensions']);
%     if exist('class','var') %doesn´t exist for multi class
%         legend(string(class), 'All');
%     else
%         legend('blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock');
%     end
% end

end
