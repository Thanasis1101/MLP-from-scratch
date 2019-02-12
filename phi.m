function result = phi(x, activation_function)

    switch activation_function
        case 'tanh'
            result = tanh(x);
        case 'sigmoid'
            result = 1./(1+exp(-x));
        case 'relu'
            result = max(0, x);
        otherwise
            result = x;
    end
    
end

