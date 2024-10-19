function perf_metric = fisher_lda(data_dev, data_test, class)

%Computes a Fisher LDA Classifier for one-vs-all classification
%and returns its perfomance metrics
%
%   Inputs:
%   data_dev -> data structure for training the classifier
%   data_test -> data structure for testing the classifier
%   class -> class to be classified against the remaining ones
%
%   Outputs:
%   per_metric -> performance metrics [sensitivity, specificity, accuracy, 
%   f1_score]

%transforming multiclass data into binary data
idx_one = find(data_dev.y==class);
idx_all = find(data_dev.y~=class);
data_dev.y(idx_one) = 1; %positive class (one)
data_dev.y(idx_all) = 2; %negative class (all)

idx_one = find(data_test.y==class);
idx_all = find(data_test.y~=class);
data_test.y(idx_one) = 1; %positive class (one)
data_test.y(idx_all) = 2; %negative class (all)

%model
model_fisher = fld(data_dev);
class_pred = linclass(data_test.X, model_fisher);

%metrics
conf_matrix = confusionmat(data_test.y, class_pred, 'Order', [1,2]);
% figure(); cm = confusionchart(data_test.y, class_pred);
perf_metric = performance(class_pred, data_test.y, conf_matrix);

% %%% PLOT
% 
% model_fisher_plot = model_fisher;
% model_fisher_plot.W = model_fisher_plot.W/norm(model_fisher_plot.W);
% model_fisher_plot.b = model_fisher_plot.b/norm(model_fisher_plot.W); 
% 
% %NÃO SEI SE SE DEVE FAZER ESTA ALTERAÇÃO NO b; comparando os plots com o do
% %euclidean, não me dá exatamente igual, oara dá parecido, ora não dá; pode
% %estar relacionado com a propria implementação interna da função; como isto
% %é só para colocar uma imagemzita no relatório, não acho que devamos perder
% %muito tempo com isto
% 
% new_data = linproj(data_dev.X, model_fisher_plot);
% data_plot = data_dev;
% data_plot.X = new_data;
% 
% idx_one = find(data_plot.y==1);
% idx_all = find(data_plot.y==2);
% 
% %prototypes
% m_one = mean(data_plot.X(:,idx_one),2);
% m_all = mean(data_plot.X(:,idx_all),2);
% 
% figure(); 
% ppatterns(data_plot);
% title('Classifier Fisher LDA')
% hold on; 
% plot(m_one, 0, '+', 'color', 'k', 'markersize', 10, 'linewidth', 2)
% plot(m_all, 0, 'o', 'color', 'k', 'markersize', 10, 'linewidth', 2)
% plot(mean([m_one m_all]), 0, 's', 'color', 'k', 'markersize', 10, 'linewidth' ,2)
% legend(['Class ' int2str(class)], 'Class All', ['Mean ' int2str(class)], 'Mean All', 'Average mean')
% hold off;

end





