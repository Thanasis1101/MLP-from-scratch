% ===== Inputs ===== %

train_set_inputs = readMNISTImages('train-images.idx3-ubyte')';
train_labels = readMNISTLabels('train-labels.idx1-ubyte');
train_set_goals = labels2goals(train_labels, 10);
test_set_inputs = readMNISTImages('t10k-images.idx3-ubyte')';
test_labels = readMNISTLabels('t10k-labels.idx1-ubyte');
test_set_goals = labels2goals(test_labels, 10);

hidden_layers_sizes = [32 16];
activation_function = 'tanh';
number_of_epochs = 10;
learning_rate = 0.01;
batch_size = 1;


% ===== Initializations ===== %

[~, number_of_hidden_layers] = size(hidden_layers_sizes);
[number_of_examples, input_layer_size] = size(train_set_inputs);
[~, output_layer_size] = size(train_set_goals);
[number_of_tests, ~] = size(test_set_inputs);

weights_and_biases = cell(number_of_hidden_layers+1, 1);
desired_weights_and_biases = cell(number_of_hidden_layers+1, 1);
for l=1:number_of_hidden_layers+1
    desired_weights_and_biases(l) = {0};
end

% randomize weights with gaussian (mean = 0, standard deviation = 1)
% random_number_of_layer = random_number / number_of_neurons_in_layer

rng(345); % set seed for reproducibility

rand_multiplier = 1 / input_layer_size;
weights_and_biases(1) = {rand_multiplier * normrnd(0, 1, [hidden_layers_sizes(1), input_layer_size+1])}; % weights and biases between input layer and first hidden layer, +1 for the bias, matrix: next_layer_size x current_layer_size
for i=2:number_of_hidden_layers
    rand_multiplier = 1 / hidden_layers_sizes(i-1);
    weights_and_biases(i) = {rand_multiplier * normrnd(0, 1, [hidden_layers_sizes(i), hidden_layers_sizes(i-1)+1])}; % weights and biases between hidden layers, +1 for the bias, matrix: next_layer_size x current_layer_size
end
rand_multiplier = 1 / hidden_layers_sizes(end);
weights_and_biases(end) = {rand_multiplier * normrnd(0, 1, [output_layer_size, hidden_layers_sizes(end)+1])}; % weights and biases between last hidden layer and output layer, +1 for the bias, matrix: next_layer_size x current_layer_size



% ===== Training ===== %


weighted_outputs = cell(number_of_hidden_layers + 1, 1); % z = w*a + ... + b
squishified_weighted_outputs = cell(number_of_hidden_layers + 1, 1); % a = phi(z)

training_errors = ones(number_of_epochs, 1);
testing_errors = ones(number_of_epochs, 1);
training_precisions = ones(number_of_epochs, 1);
testing_precisions = ones(number_of_epochs, 1);


fprintf(1,'Training...\n');
start_time = cputime;

for epoch=1:number_of_epochs
    
    sum_train_errors = 0;
    sum_test_errors = 0;
    
    % Shuffle train_set_inputs with train_set_goals %
    
    shuffler = randperm(number_of_examples);
    train_set_inputs = train_set_inputs(shuffler, :);
    train_set_goals = train_set_goals(shuffler, :);
    
    for p=1:number_of_examples

        current_example_with_bias = [train_set_inputs(p, :) 1]';
        current_goals = train_set_goals(p, :)';

        % Feed Forward (Calculation of neuron outputs) %

        z = cell2mat(weights_and_biases(1)) * current_example_with_bias;

        weighted_outputs(1) = {z};
        squishified_weighted_outputs(1) = {phi(z, activation_function)}; % outputs of neurons from the first hidden layer

        for l=2:number_of_hidden_layers+1
            z = cell2mat(weights_and_biases(l)) * [cell2mat(squishified_weighted_outputs(l-1))' 1]';
            weighted_outputs(l) = {z};
            squishified_weighted_outputs(l) = {phi(z, activation_function)};
        end   

        error = current_goals - cell2mat(squishified_weighted_outputs(end));
        sum_train_errors = sum_train_errors + sumsqr(error);
        
        % Back propagation (Calculation of desired weights) %

        sigma = phi_d(cell2mat(weighted_outputs(end)), activation_function);
        alpha = cell2mat(squishified_weighted_outputs(end-1));

        delta = sigma .* error;
        
        delta_scaled = delta(:, ones(hidden_layers_sizes(end) + 1, 1)); % scaling for every neuron of the last hidden layer
        alpha_scaled = [alpha(:, ones(output_layer_size, 1))' ones(output_layer_size, 1)]; % scaling for every neuron of the output layer
        % accumulate weights of current batch
        desired_weights_and_biases(end) = {cell2mat(desired_weights_and_biases(end)) + delta_scaled .* alpha_scaled};

        for l=number_of_hidden_layers:-1:1

            sigma = phi_d(cell2mat(weighted_outputs(l)), activation_function); 
            if l>1
                alpha = cell2mat(squishified_weighted_outputs(l-1));
            else % previous layer is input layer
                alpha = current_example_with_bias;
            end
            next_layer_weights_and_biases = cell2mat(weights_and_biases(l+1));
            next_layer_weights = next_layer_weights_and_biases(:, 1:hidden_layers_sizes(l)); % crop the last column (biases)
            previous_delta_rescaled = delta(:, ones(1, hidden_layers_sizes(l)));
            next_layer_error = sum(next_layer_weights' .* previous_delta_rescaled', 2);
                        

            delta = sigma .* next_layer_error;
            if l>1
                delta_scaled = delta(:, ones(hidden_layers_sizes(l-1) + 1, 1)); % scaling for every neuron of the current hidden layer, +1 for bias
                alpha_scaled = [alpha(:, ones(hidden_layers_sizes(l), 1))' ones(hidden_layers_sizes(l), 1)]; % scaling for every neuron of the next hidden layer
            else % previous layer is input layer
                delta_scaled = delta(:, ones(input_layer_size + 1, 1)); % scaling for every neuron of the input layer
                alpha_scaled = alpha(:, ones(hidden_layers_sizes(l), 1))'; % scaling for every neuron of the next hidden layer
            end
            % accumulate weights of current batch
            desired_weights_and_biases(l) = {cell2mat(desired_weights_and_biases(l)) + delta_scaled .* alpha_scaled};

        end

        % Update weights if we have reached the end of a batch or the end of inputs%

        if (mod(p-1, batch_size) == batch_size -1 || p==number_of_examples)
            for k=1:number_of_hidden_layers+1
                weights_and_biases(k) = {cell2mat(weights_and_biases(k)) + learning_rate .* cell2mat(desired_weights_and_biases(k))/batch_size};
                desired_weights_and_biases(k) = {0};
            end
        end


    end
    
    training_errors(epoch) = sum_train_errors / number_of_examples;
    




    % ===== Testing ===== %

    
    % Calculate precision for train set
    
    corrects = 0;

    for p=1:number_of_examples

        current_test_with_bias = [train_set_inputs(p, :) 1]';
        current_goals = train_set_goals(p, :)';

        % Feed Forward (Calculation of neuron outputs) %

        current_weighted_outputs = phi(cell2mat(weights_and_biases(1)) * current_test_with_bias, activation_function); % outputs of neurons from the first hidden layer

        for l=2:number_of_hidden_layers+1
            current_weighted_outputs = phi(cell2mat(weights_and_biases(l)) * [current_weighted_outputs; 1], activation_function);
        end

        
        [~, max_neuron_id] = max(current_weighted_outputs);
        [~, goal_neuron] = max(current_goals);
        if max_neuron_id == goal_neuron
            corrects = corrects + 1;
        end
        

    end
       
    training_precisions(epoch) = corrects/number_of_examples;
    
    
    % Calculate precision and error for test set
    
    corrects = 0;

    for p=1:number_of_tests

        current_test_with_bias = [test_set_inputs(p, :) 1]';
        current_goals = test_set_goals(p, :)';

        % Feed Forward (Calculation of neuron outputs) %

        current_weighted_outputs = phi(cell2mat(weights_and_biases(1)) * current_test_with_bias, activation_function); % outputs of neurons from the first hidden layer

        for l=2:number_of_hidden_layers+1
            current_weighted_outputs = phi(cell2mat(weights_and_biases(l)) * [current_weighted_outputs; 1], activation_function);
        end

        
        [~, max_neuron_id] = max(current_weighted_outputs);
        [~, goal_neuron] = max(current_goals);
        if max_neuron_id == goal_neuron
            corrects = corrects + 1;
        end
        
        
        error = current_goals - current_weighted_outputs;
        sum_test_errors = sum_test_errors + sumsqr(error);
        

    end
    
    testing_errors(epoch) = sum_test_errors / number_of_tests;    
    testing_precisions(epoch) = corrects/number_of_tests;
    
    %fprintf(1,'Epoch %g\n', epoch);
    %fprintf(1,'Train Error: %g\n', training_errors(epoch));
    %fprintf(1,'Train Precision: %g / %g (%g%%)\n', corrects, number_of_examples, 100 * training_precisions(epoch)); 
    %fprintf(1,'Test Error: %g\n', testing_errors(epoch));
    %fprintf(1,'Test Precision: %g / %g (%g%%)\n', corrects, number_of_tests, 100 * testing_precisions(epoch));    
    
    
end


training_time = cputime - start_time;
fprintf(1,'Seconds: %g\n', training_time); 
fprintf(1,'Precision: %g / %g (%g%%)\n', corrects, number_of_tests, 100 * testing_precisions(end));


% ===== Plot test and train precision per epoch ===== %


layers_text = num2str(input_layer_size);
for l=1:number_of_hidden_layers
    layers_text = [layers_text, 'x', num2str(hidden_layers_sizes(l))];
end
layers_text = [layers_text, 'x', num2str(output_layer_size)];
figure_title = ['nn with: layers ', layers_text, ', learning rate = ', num2str(learning_rate), ', batch size = ', num2str(batch_size), ', activation function = ', activation_function];


figure
plot(1:number_of_epochs, training_precisions);
hold on
plot(1:number_of_epochs, testing_precisions);
legend({'Train Precision','Test Precision'},'Location','southwest')
title(['Precision curves for ', figure_title]);
hold off


% ===== Plot test and train error per epoch ===== %

figure
plot(1:number_of_epochs, training_errors);
hold on
plot(1:number_of_epochs, testing_errors);
legend({'Train Error','Test Error'},'Location','southwest')
title(['Error curves for ', figure_title]);
hold off


% ===== Save the model to a file ===== %

save('nn_model.mat', 'train_set_inputs', 'train_set_goals', ...
'test_set_inputs', 'test_set_goals', 'hidden_layers_sizes', ...
'number_of_epochs', 'learning_rate', 'batch_size', ...
'activation_function', 'weights_and_biases', 'number_of_tests', ...
'training_errors', 'training_precisions', 'testing_errors', ...
'testing_precisions', 'output_layer_size', 'input_layer_size', ...
'number_of_hidden_layers');


% ===== Testing and showing some wrong predictions ===== %

corrects = 0;
wrongs_to_show = 15;
wrongs_shown = 0;
wrongs = cell(wrongs_to_show);
wrong_labels = cell(wrongs_to_show);

for p=1:number_of_tests

    current_test_with_bias = [test_set_inputs(p, :) 1]';
    current_goals = test_set_goals(p, :)';

    % Feed Forward (Calculation of neuron outputs) %

    current_weighted_outputs = phi(cell2mat(weights_and_biases(1)) * current_test_with_bias, activation_function); % outputs of neurons from the first hidden layer

    for l=2:number_of_hidden_layers+1
        current_weighted_outputs = phi(cell2mat(weights_and_biases(l)) * [current_weighted_outputs; 1], activation_function);
    end
        
    [~, max_neuron_id] = max(current_weighted_outputs);
    [~, goal_neuron] = max(current_goals);
    if max_neuron_id == goal_neuron
        corrects = corrects + 1;
    else
        % Show image
        
        if wrongs_shown < wrongs_to_show
            wrongs_shown = wrongs_shown + 1;
            wrongs(wrongs_shown) = {reshape(test_set_inputs(p, :), [28, 28])};
            wrong_labels(wrongs_shown) = {['Found: ', num2str(max_neuron_id - 1), ' Was: ', num2str(goal_neuron - 1)]};
        end
        
        
    end
    
        
end

% Show images with some of the wrongs

figure
for wrong_id=1:wrongs_to_show
    subplot(ceil(wrongs_to_show/5),5,wrong_id);
    imshow(cell2mat(wrongs(wrong_id)));
    title(cell2mat(wrong_labels(wrong_id)));
end


% ===== Show how affected every neuron is from the input (using weights) ===== %

weights_from_all_hidden_layers = 1;
for l=1:number_of_hidden_layers    
    
    figure
    for n=1:hidden_layers_sizes(l)
        neuron_weights = cell2mat(weights_and_biases(l));
        neuron_weights = neuron_weights(n,1:end-1); 
        neuron_weights = neuron_weights * weights_from_all_hidden_layers;
        neuron_weights = rescale(neuron_weights, 0, 1); % rescale to fit in range [0,1] to show it
        subplot(ceil(hidden_layers_sizes(l)/8),8,n);
        imshow(reshape(neuron_weights, [28,28]));
        title(['L', num2str(l),' neuron ', num2str(n)]);
    end
    
    % Multiply with the weights of this layer, so that the new matrix will
    % have number of columns equal to input_layer_size
    current_layer_weights = cell2mat(weights_and_biases(l));
    current_layer_weights = current_layer_weights(:,1:end-1); % crop the bias
    weights_from_all_hidden_layers = current_layer_weights * weights_from_all_hidden_layers;
    
end

% show ouput layer

figure
for n=1:output_layer_size
    neuron_weights = cell2mat(weights_and_biases(end));
    neuron_weights = neuron_weights(n,1:end-1);  % crop the bias
    neuron_weights = neuron_weights * weights_from_all_hidden_layers;
    neuron_weights = rescale(neuron_weights, 0, 1); % rescale to fit in range [0,1] to show it
    subplot(ceil(output_layer_size/5),5,n);
    imshow(reshape(neuron_weights, [28,28]));
    title(['Output neuron ', num2str(n)]);
end


% ===== Test custom input ===== %


number = rgb2gray(imread('num.jpg'));
current_test_with_bias = [reshape(rescale(double(number),0,1), [1, 28*28]) 1]';

% Feed Forward (Calculation of neuron outputs) %

current_weighted_outputs = phi(cell2mat(weights_and_biases(1)) * current_test_with_bias, activation_function); % outputs of neurons from the first hidden layer

for l=2:number_of_hidden_layers+1
    current_weighted_outputs = phi(cell2mat(weights_and_biases(l)) * [current_weighted_outputs; 1], activation_function);
end
    
[~, max_neuron_id] = max(current_weighted_outputs);

figure
imshow(number);
title(['Found: ', num2str(max_neuron_id - 1)]);


       