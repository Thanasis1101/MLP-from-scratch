function Y = rescale(X, a, b)
%RESCALE Summary of this function goes here
%   Detailed explanation goes here

    minX = min(X(:));
    maxX = max(X(:));
    Y = (b-a)*(X-minX)/(maxX-minX) + a;

end

