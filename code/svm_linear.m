function perf_metric = svm_linear(data_dev, data_test, class)

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

%c_pot = [-20:10];
c_pot = [-5:5];
C = 2.^c_pot;

n_run = 5;
error = zeros(n_run, numel(C));

for run = 1:n_run
    %for each run I divide the data_dev into data_train and data_test 50/50
    
    train = struct; validation = struct;
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
    
    for c_p = 1:numel(C)
        
        clear model
        disp(sprintf('=====\nRun = %d\nCost = %f\n', run, C(c_p)));
        if n_class == 2
            model = fitcsvm(train.X', train.y','KernelFunction', 'linear','BoxConstraint', C(c_p),'Solver', 'SMO');
        else
            svm = templateSVM('KernelFunction','linear','BoxConstraint',C(c_p),'Solver','SMO');
            model = fitcecoc(train.X',train.y','Learners',svm,'Coding','onevsone');
        end
        
        [ypred] = predict(model, validation.X');
        error(run, c_p) = cerror(ypred', validation.y)*100;
        
    end
end

err_mean = mean(error,1);
err_std = std(error,[],1);

% %%% PLOT
% figure()
% plot(c_pot, err_mean, 'o');
% ylabel('Testing Error (%)')
% set(gca, 'xtick', c_pot)
% set(gca, 'xticklabel', strcat('2^', cellfun(@num2str,num2cell(c_pot), 'UniformOutput',0)))
% hold on
% errorbar(c_pot, err_mean, err_std)

% Looking for the lower error
mean_best = min(err_mean); %encontro valor erro mínimo
idx_best = find(err_mean == mean_best); %indices de valor minimo
std_best = err_std(idx_best); %std dos valores mínimos

if length(idx_best) > 1
    [new_std,idx] = min(std_best);
    std_best = new_std;
    C_best = 2.^(c_pot(idx));
else
    C_best = 2.^(c_pot(idx_best));
end

fprintf("For SVM linear classifier, the best C is %.3f \n",C_best)
%fprintf("    - results in an error of %.2f +/- %.2f %% \n", mean_best,std_best)

%% Aplying the best model

if n_class == 2
    model = fitcsvm(data_dev.X', data_dev.y','KernelFunction', 'linear','BoxConstraint', C_best,'Solver', 'SMO');
    ypred = predict(model, data_test.X');
    conf_matrix = confusionmat(data_test.y, ypred, 'Order', [1,2]);
else
    svm = templateSVM('KernelFunction','linear','BoxConstraint',C_best,'Solver','SMO');
    model = fitcecoc(data_dev.X',data_dev.y','Learners',svm,'Coding','onevsone');
    ypred = predict(model, data_test.X');
    conf_matrix = confusionmat(data_test.y, ypred);
end

% figure()
% cm = confusionchart(data_test.y, ypred');

perf_metric = performance(ypred, data_test.y, conf_matrix);

end
