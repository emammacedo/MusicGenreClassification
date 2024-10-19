function perf_metric = svm_Nlinear(data_dev, data_test, class)

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

%C parameter
%c_pot = [-20:10];
c_pot = [-3:3];
C=2.^c_pot;

%Gama
%g_pot = [-22:0];
g_pot = [-3:0];
G = 2.^g_pot;

n_run = 5;
error = zeros(n_run, numel(C), numel(G));

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
        for g_p = 1:numel(G)
            
            clear model
            disp(sprintf('=====\nRun = %d\nCost = %f\nGama = %f\n', run, C(c_p), G(g_p)));
            if n_class == 2
                model = fitcsvm(train.X', train.y','KernelFunction', 'rbf','BoxConstraint', C(c_p),'KernelScale', sqrt(1/(2*G(g_p))), 'Solver', 'SMO');
            else
                svm = templateSVM('KernelFunction', 'rbf','BoxConstraint', C(c_p),'KernelScale', sqrt(1/(2*G(g_p))), 'Solver', 'SMO');
                model = fitcecoc(train.X',train.y','Learners',svm,'Coding', 'onevsone');
                %ver aqui questão do multiclass_method !!!!!!!!!!
            end
            
            [ypred] = predict(model, validation.X');
            error(run, c_p, g_p) = cerror(ypred', validation.y)*100;
            
        end
    end
end

err_mean = squeeze(mean(error));

% %%% PLOT
% figure(); contourf(g_pot, c_pot, err_mean)
% xlabel('Gamma')
% ylabel('Cost')
% set(gca, 'xtick', g_pot([1:5:numel(g_pot)]))
% set(gca, 'xticklabel', strcat(strcat('2^{', cellfun(@num2str, num2cell(g_pot([1:5:numel(g_pot)])), 'UniformOutput',0)),'}'))
% set(gca, 'ytick', c_pot([1:5:numel(c_pot)]))
% set(gca, 'yticklabel', strcat(strcat('2^{', cellfun(@num2str, num2cell(c_pot([1:5:numel(c_pot)])), 'UniformOutput', 0)),'}'))
% colorbar

% Looking for the lower error
[row,col] = find(err_mean == min(err_mean,[],'all'))
C_best = C(row);
G_best = G(col);

%in case of more than one pair (C,G) has the same average error, the
%function choses one
if length(row) > 1 
    C_best = C(row(1));
end
if length(col) > 1
    G_best = G(col(1));
end

fprintf("For SVM non-linear classifier, the best chosen pair (Cost, Gama) is (%.3f,%.3f) \n", C_best, G_best)


%% Aplying the best model

if n_class == 2
    model = fitcsvm(data_dev.X', data_dev.y','KernelFunction', 'rbf','BoxConstraint', C_best,'KernelScale', sqrt(1/(2*G_best)), 'Solver', 'SMO');
    ypred = predict(model, data_test.X');
    conf_matrix = confusionmat(data_test.y, ypred, 'Order', [1,2]);
else
    svm = templateSVM('KernelFunction', 'rbf','BoxConstraint', C_best,'KernelScale', sqrt(1/(2*G_best)), 'Solver', 'SMO');
    model = fitcecoc(data_dev.X',data_dev.y','Learners',svm,'Coding','onevsone');
    ypred = predict(model, data_test.X');
    conf_matrix = confusionmat(data_test.y, ypred);
end

% figure()
% cm = confusionchart(data_test.y, ypred');

perf_metric = performance(ypred, data_test.y, conf_matrix);

end