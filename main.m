%Project Music Genre Classification - Pattern Recognition
%Ema Macedo, 2019233271 | João Trovão, 2019232206
%Final deadline

% Script used to collect the results presented in the project's report. For
% each group of conditions tested, the classifiers presented in the array
% classif_names are tested. Every step of feature selection and reduction,
% treatment of test data and classifier evaluation is implemented. The 
% performance results can be consulted in the structure performance To try 
% the different classifiers in a more user-friendly way, please run the
% script GUI. 

close all, clear all, clc
rng('default'); rng(1);

%% Reading and analysing the dataset

[features_values, data_label, data_label_numeric, features_names] = read_data(); %reading the dataset
classes_perc = class_percentage(data_label); %analysing classes percentage
count = check_mis_values(features_values); %number of missing values

%% Choosing variables

method = input("One-vs-all (A) or Multiclass (B) ?\n",'s');

division_method = input("Kfold or Holdout ?\n",'s');
division_number = input("Kfold: Number of folds | HoldOut: fraction of validation data\n");
KW_number = input("Number of features selected in KW\n"); %50, 30
corr_lim = input("Correlation limit\n"); %0.85, 0.75

reduc_method = input("PCA or LDA ?\n",'s');

performance = struct;

classif_names = ["bayes", "svm_Nlinear"]; %,"bayes" "mdc_euclidian" "fisher_lda" "knn" "svm_Nlinear" "mdc_mahalanobis","bayes","svm_linear"
metrics_names = ["SS","SP","Acc","Bal_Acc","F1_score","MCC"];

for num_iter = 1:10
    
    %% Data splitting - 80% development, 20% test
    [data_dev, data_test_or] = data_splitting(features_values, data_label_numeric, features_names, 0.80);
    
    %% Quick inspection of features' values
    %Apparently, the features chroma_stft_max is equal to 1 for every sample
    
    equal = 'true';
    for i=1:size(data_dev.X,2)
        if data_dev.X(4,i) ~= 1
            equal = 'false';
            break;
        end
    end
    %fprintf("The feature has the value 1 for every sample - %s.\n",equal)
    %The feature isn't discriminative, so we are going to scrap it.
    
    if strcmp(equal,'true')
        data_dev.X(4,:) = [];
        data_dev.dim = size(data_dev.X,1);
        data_dev.feat_name(4) = [];
        data_dev.idx_og (4) = [];
    end
    
    %% Development Data standardization
    data_dev = standard(data_dev);
    
    %% Feature Selection
    [data_dev, H_values] = feat_sel_kw (data_dev, KW_number); %Kruskal-Wallis Test
    data_dev = feat_sel_corr (data_dev , H_values, corr_lim); %Correlation matrix selection
    
    %% Feature Reduction
    
    %% One-vs-All
    if strcmp(method,"A") 
        
        if strcmp(reduc_method,"PCA")
            
            data_dev_red = feat_red_pca (data_dev); %Principal Component Analysis (PCA) using Kaiser test
            
            %%% Test Data processing
            data_test = processing_test(data_test_or, data_dev);
            data_test_red = processing_test(data_test_or, data_dev_red);
            
            %%% Classifiers
            
            data = data_dev_division (data_dev, division_method, division_number);
            data_red = data_dev_division (data_dev_red, division_method, division_number);
            
            for i = 1:length(classif_names)
                for c = 1:10
                    
                    if strcmp(classif_names(i),"fisher_lda")
                        aux = classifier_testing(data, "fisher_lda", c);
                        %aux = classifier_testing({data_dev,data_test}, "fisher_lda", c);
                    else
                        aux = classifier_testing(data_red, classif_names(i), c);
                        %aux = classifier_testing({data_dev_red,data_test_red}, classif_names(i), c);
                    end
                    
                    for m = 1:length(metrics_names)
                        performance.(classif_names(i)).(metrics_names(m)) (c,num_iter) = aux{m,1};
                    end
                end
            end
            
        elseif strcmp(reduc_method,"LDA")
            
            data_test = processing_test(data_test_or, data_dev);
            
            for c = 1:10
                
                data_dev_red = feat_red_lda (data_dev, 1, c); %Linear Discriminant Analysis
                data_test_red = processing_test(data_test_or, data_dev_red);
                
                data = data_dev_division (data_dev, division_method, division_number);
                data_red = data_dev_division (data_dev_red, division_method, division_number);
                
                for i = 1:length(classif_names)
                    
                    if strcmp(classif_names(i),"fisher_lda")
                        aux = classifier_testing(data, "fisher_lda", c);
                        %aux = classifier_testing({data_dev,data_test}, "fisher_lda", c);
                    else
                        aux = classifier_testing(data_red, classif_names(i), c);
                        %aux = classifier_testing({data_dev_red,data_test_red}, classif_names(i), c);
                    end
                    
                    for m = 1:length(metrics_names)
                        performance.(classif_names(i)).(metrics_names(m)) (c,num_iter) = aux{m,1};
                    end
                end
            end
        end
        
    %% Multi classification    
    elseif strcmp(method,"B") 
        
        if strcmp(reduc_method,"PCA")
            data_dev_red = feat_red_pca (data_dev); %Principal Component Analysis (PCA) using Kaiser test
        elseif strcmp(reduc_method,"LDA")
            lda_dim = input("Dimensions to maintain in LDA\n");
            data_dev_red = feat_red_lda (data_dev, lda_dim); %Linear Discriminant Analysis
        end
        
        %%% Test Data processing
        data_test_red = processing_test(data_test_or, data_dev_red);
        
        %%% Classifiers
        data_red = data_dev_division (data_dev_red, division_method, division_number);
        
        for i = 1:length(classif_names)
            aux = classifier_testing(data_red, classif_names(i));
            %aux = classifier_testing({data_dev_red, data_test_red}, classif_names(i));
            performance.(classif_names(i)).overral (:,num_iter) = aux.overral(:,1);
            performance.(classif_names(i)).("each_class_run" + string(num_iter)) = aux.each_class;
        end
    end
end

%% Calculating average values of performance metrics

if strcmp(method,"A")
    
    for i = 1:length(classif_names)
        for m = 1:length(metrics_names)
            
            performance.(classif_names(i)).(metrics_names(m)) = array2table([mean( performance.(classif_names(i)).(metrics_names(m)) ,2), std( performance.(classif_names(i)).(metrics_names(m)),[],2)], 'RowNames',{'class 1','class 2','class 3','class 4','class 5','class 6', 'class 7', 'class 8', 'class 9', 'class 10'}, 'VariableNames',{'mean','std'});
            
        end
                    
            
    end
    

% writetable(performance.mdc_euclidian.Bal_Acc,'dados.xlsx','Sheet',1,'Range','A1')
% writetable(performance.mdc_euclidian.F1_score,'dados.xlsx','Sheet',1,'Range','C1')
% writetable(performance.mdc_euclidian.MCC,'dados.xlsx','Sheet',1,'Range','E1')
% 
% writetable(performance.mdc_mahalanobis.Bal_Acc,'dados.xlsx','Sheet',1,'Range','G1')
% writetable(performance.mdc_mahalanobis.F1_score,'dados.xlsx','Sheet',1,'Range','I1')
% writetable(performance.mdc_mahalanobis.MCC,'dados.xlsx','Sheet',1,'Range','K1')
% 
% writetable(performance.fisher_lda.Bal_Acc,'dados.xlsx','Sheet',1,'Range','M1')
% writetable(performance.fisher_lda.F1_score,'dados.xlsx','Sheet',1,'Range','O1')
% writetable(performance.fisher_lda.MCC,'dados.xlsx','Sheet',1,'Range','Q1')
% 
% writetable(performance.knn.Bal_Acc,'dados.xlsx','Sheet',1,'Range','S1')
% writetable(performance.knn.F1_score,'dados.xlsx','Sheet',1,'Range','U1')
% writetable(performance.knn.MCC,'dados.xlsx','Sheet',1,'Range','W1')
% 
% writetable(performance.bayes.Bal_Acc,'dados.xlsx','Sheet',1,'Range','Y1')
% writetable(performance.bayes.F1_score,'dados.xlsx','Sheet',1,'Range','AA1')
% writetable(performance.bayes.MCC,'dados.xlsx','Sheet',1,'Range','AC1')
% 
% writetable(performance.svm_linear.Bal_Acc,'dados.xlsx','Sheet',1,'Range','AE1')
% writetable(performance.svm_linear.F1_score,'dados.xlsx','Sheet',1,'Range','AG1')
% writetable(performance.svm_linear.MCC,'dados.xlsx','Sheet',1,'Range','AI1')
% 
% writetable(performance.svm_Nlinear.Bal_Acc,'dados.xlsx','Sheet',1,'Range','AK1')
% writetable(performance.svm_Nlinear.F1_score,'dados.xlsx','Sheet',1,'Range','AM1')
% writetable(performance.svm_Nlinear.MCC,'dados.xlsx','Sheet',1,'Range','AO1')

elseif strcmp(method,"B")
    
    for i = 1:length(classif_names)
        aux = performance.(classif_names(i)).overral {:,:};
        
        performance.(classif_names(i)).overral = [];
        performance.(classif_names(i)).overral (:,1) = mean(aux,2);
        performance.(classif_names(i)).overral (:,2) = std(aux,[],2);
        
        performance.(classif_names(i)).overral = array2table([mean(aux,2), std(aux,[],2)], 'RowNames',{'SS','SP','Acc','F1 score','MCC'}, 'VariableNames',{'mean','std'});
        
        soma = 0;
        for j = 1:num_iter
            soma = soma + performance.(classif_names(i)).("each_class_run" + string(j)) {:,:} ;
            performance.(classif_names(i)) = rmfield(performance.(classif_names(i)),"each_class_run" + string(j))
        end
        
        performance.(classif_names(i)).each_class = array2table(soma/num_iter, 'VariableNames',{'SS','SP','Acc','Bal Acc','F1 score', 'MCC','Precision'}, 'RowNames', {'class 1','class 2','class 3','class 4','class 5','class 6', 'class 7', 'class 8', 'class 9', 'class 10'});
    end
end

% writetable(performance.mdc_euclidian.overral,'dados.xlsx','Sheet',1,'Range','B7')
% writetable(performance.mdc_euclidian.each_class,'dados.xlsx','Sheet',1,'Range','D7')

% writetable(performance.bayes.overral,'dados.xlsx','Sheet',1,'Range','K7')
% writetable(performance.bayes.each_class,'dados.xlsx','Sheet',1,'Range','M7')

% writetable(performance.svm_linear.overral,'dados.xlsx','Sheet',1,'Range','T7')
% writetable(performance.svm_linear.each_class,'dados.xlsx','Sheet',1,'Range','V7')

% writetable(performance.svm_Nlinear.overral,'dados.xlsx','Sheet',1,'Range','T7')
% writetable(performance.svm_Nlinear.each_class,'dados.xlsx','Sheet',1,'Range','V7')


