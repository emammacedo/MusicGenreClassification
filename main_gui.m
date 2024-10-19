function performance = main_gui(percentage_dev, KW, KW_number, corr_lim, reduc_method, lda_dim, division_method, division_number, method, classif_names, c)

%Project Music Genre Classification - Pattern Recognition
%Ema Macedo, 2019233271 | João Trovão, 2019232206
%Final deadline

% Script used by the GUI


%% Reading and analysing the dataset

[features_values, data_label, data_label_numeric, features_names] = read_data(); %reading the dataset
classes_perc = class_percentage(data_label); %analysing classes percentage
count = check_mis_values(features_values); %number of missing values

%% Choosing variables


performance = struct;

metrics_names = ["SS","SP","Acc","Bal_Acc","F1_score","MCC"];

for num_iter = 1:5
    
    %% Data splitting - 80% development, 20% test
    [data_dev, data_test_or] = data_splitting(features_values, data_label_numeric, features_names, percentage_dev);
    
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
    if KW==1
        [data_dev, H_values] = feat_sel_kw (data_dev, KW_number); %Kruskal-Wallis Test
        data_dev = feat_sel_corr (data_dev , H_values, corr_lim); %Correlation matrix selection
    end
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
     
                if strcmp(classif_names(i),"fisher_lda")
                    aux = classifier_testing(data, "fisher_lda", c);
                else
                    aux = classifier_testing(data_red, classif_names(i), c);
                end

                for m = 1:length(metrics_names)
                    performance.(classif_names(i)).(metrics_names(m)) (c,num_iter) = aux{m,1};
                end
                
            end
            
        elseif strcmp(reduc_method,"LDA")
            
            data_test = processing_test(data_test_or, data_dev);
       
            data_dev_red = feat_red_lda (data_dev, 1, c); %Linear Discriminant Analysis
            data_test_red = processing_test(data_test_or, data_dev_red);

            data = data_dev_division (data_dev, division_method, division_number);
            data_red = data_dev_division (data_dev_red, division_method, division_number);

            for i = 1:length(classif_names)

                if strcmp(classif_names(i),"fisher_lda")
                    aux = classifier_testing(data, "fisher_lda", c);
                else
                    aux = classifier_testing(data_red, classif_names(i), c);
                end

                for m = 1:length(metrics_names)
                    performance.(classif_names(i)).(metrics_names(m)) (c,num_iter) = aux{m,1};
                end
            end
        end
        
    %% Multi classification    
    elseif strcmp(method,"B") 
        
        if strcmp(reduc_method,"PCA")
            data_dev_red = feat_red_pca (data_dev); %Principal Component Analysis (PCA) using Kaiser test
        elseif strcmp(reduc_method,"LDA")
            data_dev_red = feat_red_lda (data_dev, lda_dim); %Linear Discriminant Analysis
        end
        
        %%% Test Data processing
        data_test_red = processing_test(data_test_or, data_dev_red);
        
        %%% Classifiers
        data_red = data_dev_division (data_dev_red, division_method, division_number);
        
        for i = 1:length(classif_names)
            aux = classifier_testing(data_red, classif_names(i));
            performance.(classif_names(i)).overral (:,num_iter) = aux.overral(:,1);
            performance.(classif_names(i)).("each_class_run" + string(num_iter)) = aux.each_class;
        end
    end
end

%% Calculating average values of performance metrics

if strcmp(method,"A")
    
    for i = 1:length(classif_names)
        for m = 1:length(metrics_names)
            
            %performance.(classif_names(i)).(metrics_names(m)) = array2table([mean( performance.(classif_names(i)).(metrics_names(m)) ,2), std( performance.(classif_names(i)).(metrics_names(m)),[],2)], 'RowNames',{'class 1','class 2','class 3','class 4','class 5','class 6', 'class 7', 'class 8', 'class 9', 'class 10'}, 'VariableNames',{'mean','std'});
            
        end
                    
            
    end
    

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


end

