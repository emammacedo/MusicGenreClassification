function data = data_dev_division (data_dev, type, iter)

train.dim = data_dev.dim;

if isfield(data_dev, 'pca_model') == 1 %pca
    train.pca_model = data_dev.pca_model;
elseif isfield(data_dev, 'lda_model') == 1 %lda
    train.lda_model = data_dev.lda_model;
end

data = {};

if isequal(type,"Kfold")
    %iter = number of folds
    ind = crossvalind('Kfold',data_dev.num_data,iter)';
    
    for i=1:iter %setting up the datasets
  
    ind_test = find(ind == i);
    test.X = data_dev.X(:,ind_test);
    test.y = data_dev.y(ind_test);
    test.num_data = length(test.y);
    
    ind_train = find(ind ~= i);
    train.X = data_dev.X(:,ind_train);
    train.y = data_dev.y(ind_train);
    train.num_data = length(train.y);
    
    data{i,1} = train;
    data{i,2} = test;
    
    end

elseif isequal(type,"HoldOut")
    %iter = percentage of data to test
    
    for i=1:10
        
        ind = crossvalind('HoldOut',data_dev.num_data,iter)';
        
        ind_test = find(ind == 0);
        test.X = data_dev.X(:,ind_test);
        test.y = data_dev.y(ind_test);
        test.num_data = length(test.y);
        
        ind_train = find(ind == 1);
        train.X = data_dev.X(:,ind_train);
        train.y = data_dev.y(ind_train);
        train.num_data = length(train.y);
        
        data{i,1} = train;
        data{i,2} = test;
        
    end
    
end

end