function result = phi_d(x, activation_function)

    switch activation_function
        case 'tanh'
            result =  1 - tanh(x).^2;
        case 'sigmoid'
            result = (exp(-x)) ./ ((1+exp(-x)).^2);
        case 'relu'
            if x<0
                result = 0;
            else
                result = 1;
            end
        otherwise
            result = x;
    end
    
end

