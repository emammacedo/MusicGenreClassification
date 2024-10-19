%function [data_scaled, datat_scaled]=scalestd(data)%,data_t)
function data_scaled = scalestd(data)
%Scale data in order to have zero mean and unitary standard deviation

mu=mean(data.X,2);
s=std(data.X,[],2);

data_scaled=data;
data_scaled.X=(data.X-repmat(mu,1,data.num_data))./repmat(s,1,data.num_data);

% datat_scaled=data_t;
% datat_scaled.X=(data_t.X-repmat(mu,1,data_t.num_data))./repmat(s,1,data_t.num_data);

end