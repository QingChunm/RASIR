   
load('Q.mat');
load('V.mat');
f = fspecial('gaussian',[11 11]);
f = f(:);
for type = 1 : 4
    for idx = 1 : 216
    % solve the regression approximately
        if(V(:,type,idx)==0)
            filters(:, type, idx) = f;
        else
            filters(:, type, idx) = Q(:, :, type, idx) \ V(:, type, idx);
        end;
    end
end
save('filters2.mat', 'filters');