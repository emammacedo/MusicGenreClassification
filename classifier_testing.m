function metrics = classifier_testing (data, classifier, class)

num_iter = size(data,1);

if nargin == 3 %binary
    aux = [];
    % aux -> [num_folds x num_metrics]
    
    for i=1:num_iter
        
        train = data{i,1};
        test = data{i,2};
        
        if isequal(classifier,"mdc_euclidian")
            aux = [aux ; mdc_euclidian(train, test, class)];
        elseif isequal(classifier,"mdc_mahalanobis")
            aux = [aux ; mdc_mahalanobis(train, test, class)];
        elseif isequal(classifier,"bayes")
            aux = [aux ; bayes(train, test, class)];
        elseif isequal(classifier,"knn")
            aux = [aux ; knn(train, test, class)];
        elseif isequal(classifier,"svm_linear")
            aux = [aux ; svm_linear(train, test, class)];
        elseif isequal(classifier,"svm_Nlinear")
            aux = [aux ; svm_linear(train, test, class)];
        elseif isequal(classifier,"fisher_lda")
            aux = [aux ; fisher_lda(train, test, class)];
        end
        
    end
    
    metrics = array2table([mean(aux,1); std(aux,[],1)]', 'RowNames',{'SS','SP','Acc','Bal Acc','F1 score', 'MCC'}, 'VariableNames',{'mean','std'});
    
else %multi class
    aux = [];
    
    for i=1:num_iter
        
        train = data{i,1};
        test = data{i,2};
        
        if isequal(classifier,"mdc_euclidian")
            aux = [aux ; mdc_euclidian(train, test)];
        elseif isequal(classifier,"mdc_mahalanobis")
            aux = [aux ; mdc_mahalanobis(train, test)];
        elseif isequal(classifier,"bayes")
            aux = [aux ; bayes(train, test)];
        elseif isequal(classifier,"knn")
            aux = [aux ; knn(train, test)];
        elseif isequal(classifier,"svm_linear")
            aux = [aux ; svm_linear(train, test)];
        elseif isequal(classifier,"svm_Nlinear")
            aux = [aux ; svm_Nlinear(train, test)];
        end
    end
    
    
    aux_ec = aux(1).each_class;
    aux_o = aux(1).overral;
    
    for i=2:num_iter
        aux_ec = aux_ec + aux(i).each_class;
        aux_o = [aux_o ; aux(i).overral];
    end
    
    aux_ec = aux_ec/num_iter;
    
    metrics.each_class = array2table(aux_ec,'VariableNames',{'SS','SP','Acc','Bal Acc','F1 score', 'MCC','Precision'}, 'RowNames', {'class 1','class 2','class 3','class 4','class 5','class 6', 'class 7', 'class 8', 'class 9', 'class 10'});
    metrics.overral = array2table([mean(aux_o,1); std(aux_o,[],1)]', 'RowNames',{'SS','SP','Acc','F1 score', 'MCC'}, 'VariableNames',{'mean','std'});
    
end

end