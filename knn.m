function perf_metric = knn(data_dev, data_test, class)

%Computes a KNN Classifier for one-vs-all and multi class classification 
%and returns its performance metrics. It chooses automatically the best k
%value (with a maximum of k = 30)
%
%   Inputs:
%   data_dev -> data structure for training the classifier
%   data_test -> data structure for testing the classifier
%   class -> class to be classified against the remaining ones (only for
%   one-vs-all)
%
%   Outputs:
%   perf_metric -> performance metrics

%data_test é para testar o modelo depois do K escolhido
%data_dev é partido em vários para as diferentes runs

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

n_run = 10; %30
n_k = 30;
error = zeros(n_run, n_k);

for run = 1:n_run 
    %for each run I divide the data_dev into data_train and data_test 50/50
    
    %train = struct; validation = struct;
    shuffle = randperm(data_dev.num_data);
    idx_sep = floor(data_dev.num_data/2); 

    train.X = data_dev.X(:,shuffle(1:idx_sep));
    train.y = data_dev.y(shuffle(1:idx_sep),1);
    train.dim = size(train.X,1);
    train.num_data = size(train.X,2);
    
    validation.X = data_dev.X(:,shuffle(idx_sep+1:end));
    validation.y = data_dev.y(shuffle(idx_sep+1:end),1);
    validation.dim = size(validation.X,1);
    validation.num_data = size(validation.X,2);
    
    for k = 1:n_k
        
        model = knnrule(train, k);
        ypred = knnclass(validation.X, model);
        error(run,k) = cerror(ypred',validation.y)*100;
        
    end
end

err_med = mean (error,1);
err_std = std(error,[],1);

% %%% PLOT
% figure()
% errorbar(err_med,err_std)
% title('Error as a function of k')

%best k 
mean_k_best = min(err_med);
k_best = find(err_med == mean_k_best);
std_k_best = err_std(k_best);

if length(k_best) > 1
     new_std = min(std_k_best);
     std_k_best = new_std;
     k_best = k_best(find(std_k_best == new_std));
end

fprintf("For KNN classifier, the best k is %.0f \n",k_best)
fprintf("    - results in an error of %.2f +/- %.2f %% \n",mean_k_best,std_k_best)

%knn classifier
model = knnrule(data_dev, k_best);
ypred = knnclass(data_test.X, model);

if  n_class == 2
    conf_matrix = confusionmat(data_test.y, ypred, 'Order', [1,2]);
else
    conf_matrix = confusionmat(data_test.y, ypred);
end

% figure()
% cm = confusionchart(data_test.y, ypred');

perf_metric = performance(ypred, data_test.y, conf_matrix);

end