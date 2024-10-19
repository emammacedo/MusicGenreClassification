function perf_metric = mdc_mahalanobis(data_dev, data_test, class)

%Computes an Mahalanobis Minimum Distance Classifier for one-vs-all and multi 
%class classification and returns its performance metrics. It allows to 
%plot the classifier for one-vs-all classif.
%
%   Inputs:
%   data_dev -> data structure for training the classifier
%   data_test -> data structure for testing the classifier
%   class -> class to be classified against the remaining ones (only for
%   one-vs-all)
%
%   Outputs:
%   perf_metric -> performance metrics

if nargin == 3 %for binary classification, we specify wich class we want
    
    if isfield(data_dev, 'pca_model') == 1 %pca
        idx_one = find(data_dev.y == class);
        idx_all = find(data_dev.y ~= class);
        data_dev.y(idx_one) = 1;
        data_dev.y(idx_all) = 2;
        
        idx_one = find(data_test.y==class);
        idx_all = find(data_test.y~=class);
        data_test.y(idx_one) = 1;
        data_test.y(idx_all) = 2;
    end
    
    %finding samples of each class
    idx_one = find(data_dev.y==1);
    idx_all = find(data_dev.y==2);
    
    %prototypes
    m_one = mean(data_dev.X(:,idx_one),2);
    m_all = mean(data_dev.X(:,idx_all),2);
    
    %covariance matrix for both classes
    C_one = cov(data_dev.X(:,idx_one)');
    C_all = cov(data_dev.X(:,idx_all)');
    
    C = (C_one+C_all)./2; %pooled covariance
    C_inv = inv(C);
    
    %classification of samples
    class_pred = zeros(1,data_test.num_data);
    for i=1:data_test.num_data
        g_one = m_one'*C_inv*data_test.X(:,i)-0.5*m_one'*C_inv*m_one;
        g_all = m_all'*C_inv*data_test.X(:,i)-0.5*m_all'*C_inv*m_all;
        if g_one>=g_all
            class_pred(i) = 1;
        else
            class_pred(i) = 2;
        end
    end
    
    conf_matrix = confusionmat(data_test.y, class_pred, 'Order', [1,2]);
    
%     %%% PLOT
%     
%     data_dev.y(find(data_dev.y == 1)) = 0; 
%     data_dev.y(find(data_dev.y == 2)) = 1;
%     
%     w = (m_one'-m_all')*C_inv;
%     %w = ((m_one'-m_all')*C_inv)';
%     b = -1/2*(m_one'*C_inv*m_one-m_all'*C_inv*m_all);
%     
%     figure()
%     ppatterns(data_dev);
%     title('Classifier MDC Mahalanobis')
%     xlabel('PC1')
%     ylabel('PC2')
%     hold on
%     
%     if data_dev.dim == 1
%         plot(m_one, 0, '+', 'color', 'r', 'markersize', 10, 'linewidth', 2)
%         plot(m_all, 0, 'o', 'color', 'r', 'markersize', 10, 'linewidth', 2)
%         plot(mean([m_one m_all]), 0, 's', 'color', 'r', 'markersize', 10, 'linewidth' ,2)
%         legend(['Class ' int2str(class)], 'Class All', ['Mean ' int2str(class)], 'Mean All', 'Average mean')
%     elseif data_dev.dim == 2
%         plot(m_one(1), m_one(2), '+', 'color', 'r', 'markersize', 10, 'linewidth', 2)
%         plot(m_all(1), m_all(2), 'o', 'color', 'r', 'markersize', 10, 'linewidth', 2)
%         plot((m_one(1)+m_all(1))/2,(m_one(2)+m_all(2))/2, 's', 'color', 'r', 'markersize', 10, 'linewidth' ,2)
%         line([m_one(1),m_all(1)], [m_one(2),m_all(2)] , 'linestyle' ,'--', 'color','k','linewidth',1)
%         pline(w,b,'r')
%         legend('Class All',['Class ' int2str(class)],['Mean ' int2str(class)],'Mean All','Average Mean', 'Line between means','Hyperplane')
%     elseif data_dev.dim == 3
%         plot3(m_one(1), m_one(2), m_one(3), '+', 'color', 'r', 'markersize', 10, 'linewidth', 2)
%         plot3(m_all(1), m_all(2), m_all(3), 'o', 'color', 'r', 'markersize', 10, 'linewidth', 2)
%         plane3(w,b)
%         zlabel('PC3')
%         legend(['Class ' int2str(class)], 'Class All',['Mean ' int2str(class)],'Mean All','Hyperplane')
%     end
%     hold off
    
else %multi class classification - 10 classes
    
    m = []; %indice corresponde á classe
    c = zeros(data_dev.dim,data_dev.dim);
    
    for i = 1 : 10
        aux = find(data_dev.y==i);
        mean_aux = mean(data_dev.X(:,aux),2);
        m = [m mean_aux];
        cov_aux = cov(data_dev.X(:,aux)');
        c = c + cov_aux;
    end
    
    c = c./2;
    C_inv = inv(c);
    
    class_pred = zeros(1,data_test.num_data);
    
    g = [];
    for i=1:data_test.num_data
        for j = 1:10
            g_aux = m(:,j)'*C_inv*data_test.X(:,i)-0.5*m(:,j)'*C_inv*m(:,j);
            g = [g g_aux];
        end
        [M,I] = max(g);
        if length (I) == 1
            class_pred(i) = I;
        else
            class_pred(i) = I(1);
        end
        g = [];
    end
    
    conf_matrix = confusionmat(data_test.y, class_pred');
    
end

% figure()
% cm = confusionchart(data_test.y, class_pred)

perf_metric = performance(class_pred, data_test.y, conf_matrix);

end