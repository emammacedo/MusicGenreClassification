function count = check_mis_values (features_values)

%Checks if there's missing values in teh dataser
%
%   Inputs:
%   features_values -> matrix with features values
%
%   Outputs:
%   count -> number of missing values

count = 0;
for i=1:size(features_values,1)
    for j=1:size(features_values,2)
        if isnan(features_values(i,j))==1
            count = +1;
        end
    end
end
%fprintf ("Number of NaN values = %d\n",count)
end