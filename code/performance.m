function metrics = performance (class_pred, class_true, conf_matrix)

%Evaluates classifier performance, using a group of metrics
%
%   Inputs:
%   class_pred -> predicted labels
%   class_true -> true labels
%   conf_matrix -> confusion matrix
%
%   Outputs:
%   for binary classification, metrics is a matrix
%   for multiclass classification, metrics is a struct


n_class = size(conf_matrix,1); %number of classes

accuracy = (1 - cerror(class_pred, class_true))*100;



if n_class == 2 %one-vs-all
    
    TP = conf_matrix(1,1);
    TN = conf_matrix(2,2);
    FP = conf_matrix(1,2);
    FN = conf_matrix(2,1);
    
    if TP + FN == 0
        ss = 0;
    else
        ss = TP/(TP + FN)*100;
    end
    
    sp = TN/(TN + FP)*100;
    
    bal_accuracy = (ss + sp)/2;
    mcc = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
    
    if TP +FP == 0
        precision = 0 ;
    else
        precision = TP/(TP + FP);
    end
    
    if TP + FN == 0
        recall = 0 ;
    else
        recall = TP/(TP + FN);
    end
    
    f1_score = 2*precision*recall / (precision +  recall);
    
    
    metrics = [ss, sp, accuracy, bal_accuracy, f1_score, mcc];
    %--> analisar bal_accuracy, f1_score, mcc (boas métricas para
    %unbalanced data)
    
else %multi classification
    
    metrics = struct;
    TP = []; TN = []; FP = []; FN = [];
    
    for i = 1:n_class
        TP(i) = conf_matrix(i,i);
        FP(i) = sum(conf_matrix(:,i)) - conf_matrix(i,i);
        FN(i) = sum(conf_matrix(i,:)) - conf_matrix(i,i);
        TN(i) = sum(sum(conf_matrix)) - FP(i) - FN(i) - TP(i);
    end
    
    ss = zeros(1,n_class);  sp = ss;
    f1_score = zeros(1,n_class);
    acc = zeros(1,n_class); bal_acc = acc;
    mcc = zeros(1,n_class);
    precision = zeros(1,n_class); recall = precision;
    pk = zeros(1,n_class); tk = pk;
    
    for i = 1:n_class
        
        if TP(i) + FN(i) == 0
            ss(i) = 0;
        else
            ss(i) = TP(i)/(TP(i) + FN(i))*100;
        end
        %ss(i) = TP(i)/(TP(i) + FN(i))*100;
        sp(i) = TN(i)/(TN(i) + FP(i))*100;
        
        if TP(i) + FP(i) == 0
            precision(i) = 0 ;
        else
            precision (i) = TP(i)/(TP(i) + FP(i));
        end
        
        if TP(i) + FN(i) == 0
            recall(i) = 0 ;
        else
            recall (i) = TP(i)/(TP(i) + FN(i));
        end
        
        %precision (i) = TP(i)/(TP(i) + FP(i));
        %recall (i) = TP(i)/(TP(i) + FN(i));
        f1_score(i) = 2*precision(i)*recall(i) / (precision(i) +  recall(i)) * 100;
        
        acc(i) = (TP(i) + TN(i))/(TP(i)+ TN(i) + FP(i) + FN(i))*100;
        bal_acc(i) = (ss(i) + sp(i))/2;
        
        mcc(i) = (TP(i)*TN(i) - FP(i)*FN(i)) / sqrt((TP(i)+FP(i))*(TP(i)+FN(i))*(TN(i)+FP(i))*(TN(i)+FN(i)));
        
        pk (i) = sum(conf_matrix(:,i));
        tk (i) = sum(conf_matrix(i,:));
    end
    
    metrics.each_class = [ss' , sp',  acc', bal_acc', f1_score', mcc', precision'];
    
    ss_m = mean(ss);
    sp_m = mean(sp);
    macro_f1_score = 2* ( (mean(precision) * mean(recall)) / ( mean(precision)^(-1) + mean(recall)^(-1) ));
    mcc_multiclass = (sum(TP) * sum(sum(conf_matrix)) - sum(pk.*tk)) / sqrt(( sum(sum(conf_matrix))^2 - sum(pk.^2))*( sum(sum(conf_matrix))^2 - sum(tk.^2)));

    metrics.overral = [ss_m, sp_m, accuracy, macro_f1_score, mcc_multiclass];
    
end

end

%Para problemas com dados não balenceada --> problemas binários
%Precision = TP / (TP + FP)
%Recall = TP / (TP + FN)
%F1Score = 2*Precision*Recall / (Precision +  Recall)