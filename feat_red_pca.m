function data_pca = feat_red_pca(data_dev)

%Performs PCA and dimension reduction to the adequate number of Principal
%Components using Kaiser criterion
%
%   Inputs:
%   data_dev -> data structure
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .feat_name [1 x dim] -> names of the features
%       .idx_og [1 x dim] -> indexes of the features in the initial dataset
%       .st [2 x dim]-> meand and variance of each feature
%
%   Outputs:
%   data_pca -> data structure with selected principal components
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .feat_name [1 x num_feat] -> names of the features
%       .idx_og [1 x num_feat] -> indexes of the features in the initial dataset
%       .st [2 x dim]-> meand and variance of each feature
%       .pca_model [struct]-> pca model (to be applied to test data)

model = pca(data_dev.X); 

% %plot Eigen values
% figure(); stem(model.eigval)
% xlabel('Principal Component'); ylabel('Eig. Value');
% title('Eigenvalues'); grid on
% 
% %plot % of preserved variance
% % total_variance=sum(model.eigval)
% % fprintf('Total variance of the data (PCA): %f', total_variance);
% figure() ;plot(cumsum(model.eigval.^2)./sum(model.eigval.^2)*100,'o')
% xlabel('Principal Component'); ylabel('% of variance')
% title('Cumulative Variance (%)'); grid on

%Kaiser criterion
eign_values = model.eigval;
dim=length(eign_values(eign_values>=1));

model_reduction=pca(data_dev.X, dim); 

data_pca=data_dev;
data_pca.X=linproj(data_dev.X,model_reduction);
data_pca.dim=dim;
data_pca.pca_model = model_reduction;

% %plot
% if dim<=3
%     figure()
%     ppatterns(data_pca)
%     xlabel('PC1')
%     ylabel('PC2')
%     if dim == 3
%         zlabel('PC3')
%     end
%     title(['PCA - ' num2str(dim) ' dimensions']);
%     legend('blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock');
end
