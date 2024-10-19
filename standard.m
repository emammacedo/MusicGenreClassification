function data_st = standard (data)

%Performs data standardization
%
%   Inputs:
%   data -> data structure
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%
%   Outputs:
%   data_st -> data structure with standardized data
%       .X [dim x num_data]
%       .y [1 x num_data] -> numeric labels
%       .st [2 x dim]-> meand and variance of each feature

m=mean(data.X,2); %mean along the rows
s=std(data.X,[],2); %variance along the rows

data_st=data;
for i=1:data.dim
    for j=1:data.num_data
        data_st.X(i,j) = (data.X(i,j)-m(i,1))/s(i,1);
    end
end
data_st.st=[m';s'];
end