function perf_metric = bayes(data_dev, data_test, class)

%Computes a Bayes Classifier for one-vs-all and multi class classification 
%and returns its performance metrics. It allows to plot the classifier for 
%2 dimensions.
%
%   Inputs:
%   data_dev -> data structure for training the classifier
%   data_test -> data structure for testing the classifier
%   class -> class to be classified against the remaining ones (only for
%   one-vs-all)
%
%   Outputs:
%   perf_metric -> performance metrics

if nargin == 3
    n_class = 2;
    
    if isfield(data_dev, 'pca_model') == 1 %pca
        
        idx_one = find(data_dev.y==class);
        idx_all = find(data_dev.y~=class);
        data_dev.y(idx_one) = 1; %positive class (one)
        data_dev.y(idx_all) = 2; %negative class (all)
        
        idx_one = find(data_test.y==class);
        idx_all = find(data_test.y~=class);
        data_test.y(idx_one) = 1; %positive class (one)
        data_test.y(idx_all) = 2; %negative class (all)
    end
else
    n_class = 10;
end

model = struct;
model.Prior = [];

for i = 1:n_class
    
    idx_aux = find(data_dev.y==i);
    
    dev_aux = data_dev;
    dev_aux.X = data_dev.X(:,idx_aux);
    dev_aux.y = data_dev.y(idx_aux);
    dev_aux.num_data = numel(idx_aux);
    
    model.Pclass{i} = mlcgmm(dev_aux); %Estimate Likelihood for class i = p(X|wi)
    
    P_aux = length(idx_aux)/length(data_dev.y); %Estimate Prior probability for class i = P(wi)
    
    model.Prior = [model.Prior P_aux];
    model.fun = 'bayescls';
    
end

ypred = bayescls(data_test.X, model); %0/1 loss function 0-loss for TP & TN and equal loss for wrong classif

if  n_class == 2
    conf_matrix = confusionmat(data_test.y, ypred, 'Order', [1,2]);
else
    conf_matrix = confusionmat(data_test.y, ypred);
end

% figure()
% cm = confusionchart(data_test.y, ypred);

perf_metric = performance(data_test.y, ypred, conf_matrix);

% %%% PLOT
% if data_dev.dim <= 2
%     figure()
%     ppatterns(data_dev);
%     pboundary(model);
%     title('Bayes Classifier');
%     if  n_class == 2
%         legend(['Class ' int2str(class)], 'Class All');
%     else
%         legend('blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock');
%     end
% end

end