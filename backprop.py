from random import seed, uniform, randrange, shuffle
from math import exp, tanh, sinh, cosh, isnan
from csv import reader
from copy import deepcopy

# Function to load data from input file
def load_data_from_file(filename, data_class):
    dataset = list()
    with open(filename, "r") as input_file:
        for line in input_file:
            new_line = line.strip().split(",")
            # Append class of data (ie the digit that the data represents)
            new_line.append(data_class) 
            dataset.append(new_line)
    return dataset

# Function to convert strings in list to floats
def convert_list_to_floats(dataset):
    for line in dataset:
        for i in range(len(line) - 1):
            line[i] = float(line[i])
    return dataset

# Function to do k-fold split of dataset
def crossval_split(dataset, k_folds):
    k_split_dataset = list()
    copy_of_dataset = list(dataset)
    fold_size = int(len(dataset) / k_folds)
    for i in range(k_folds):
        fold = list()
        for j in range(fold_size):
            # Select random data element from dataset and add to current fold
            index = randrange(len(copy_of_dataset))
            fold.append(copy_of_dataset.pop(index))
        k_split_dataset.append(fold)
    return k_split_dataset

# Function to holdout a subsection of the training data as validation data
def holdout_training_data(training_data, validation_split):
    validation_data = list()
    validation_size = int(len(training_data) * validation_split)
    for i in range(validation_size):
        # Select random data element from training data and add to validation data
        index = randrange(len(training_data))
        validation_data.append(training_data.pop(index))
    return training_data, validation_data

# Driving function of the neural network which prompts initialization, training, and prediction
def do_neural_net(training_data, testing_data, batch_size, learn_rate, momentum, num_epochs, num_hidden_neurons, actv_func):
    # Get number of inputs per line and number of lines in training data
    num_inputs = len(training_data[0]) - 1
    num_outputs = len(set([line[-1] for line in training_data]))
    # Initialize network
    network = init_network(num_inputs, num_hidden_neurons, num_outputs)
    # Start training
    train(network, training_data, batch_size, learn_rate, momentum, num_epochs, num_outputs, actv_func)
    predictions = make_predictions(network, testing_data, actv_func)
    return predictions, network

# Function to initialize the network
def init_network(num_inputs, num_hidden_neurons, num_outputs):
    network = list()
    # Create hidden and output layers as sets containing nuerons represented as dictionaries (allows separation of weights, bias, and error)
    # There are num_input + 1 weights because the last "weight" is the bias 
    hidden_layer = list()
    for i in range(num_hidden_neurons):
        neuron = {"w": [uniform(-1.0, 1.0) for i in range(num_inputs + 1)]}
        hidden_layer.append(neuron)
    output_layer = list()
    for i in range(num_outputs):
        neuron = {"w": [uniform(-1.0, 1.0) for i in range(num_hidden_neurons + 1)]}
        output_layer.append(neuron)
    network.append(hidden_layer)
    network.append(output_layer)
    return network

# Function for training the network
def train(network, training_data, batch_size, learn_rate, momentum, num_epochs, num_outputs, actv_func):
    # Contains the change in weight for each weight of each neuron; used to calculate momentum factor
    weight_deltas = list()
    # Build list in same shape as network
    for layer in network:
        d_layer = list()
        for neuron in layer:
            d_neuron = list()
            for i in range(len(neuron["w"]) - 1):
                d_neuron.append(0)
            d_layer.append(d_neuron)
        weight_deltas.append(d_layer)
    
    for epoch in range(num_epochs):
        sum_error = 0
        shuffle(training_data)
        curr_line = 0
        # Check if training data divides evenly into batches
        overflow = len(training_data) % batch_size  
        for line in training_data:
            curr_line += 1
            # Propagate input forward
            outputs = forward_prop(network, line, actv_func)
            # What we want the neural network to output
            target = [0 for i in range(num_outputs)]
            # line[-1] is the classification of the input (ie the digit it represents), so we set the target to the output being a 100% confident prediction for that class
            target[line[-1]] = 1
            # Get sum of errors
            for i in range(len(target)):
                sum_error += (target[i] - outputs[i]) ** 2
            # Propagate error backwards
            back_prop_error(network, target, actv_func)
            if(overflow == 0): # If no overflow:
                if(curr_line % batch_size == 0): # Update at end of each batch
                    # Copy of network used to update weight deltas
                    network_prev_batch = deepcopy(network)
                    # Update weights
                    update_weights(network, line, learn_rate, momentum, weight_deltas)
                    weight_deltas = update_weight_deltas(network_prev_batch, network, weight_deltas)
            else: # If overflow:
                # Update at end of each batch OR end of training data
                if(curr_line % batch_size == 0 or curr_line == len(training_data)):
                    # Copy of network used to update weight deltas
                    network_prev_batch = deepcopy(network)
                    # Update weights
                    update_weights(network, line, learn_rate, momentum, weight_deltas)
                    weight_deltas = update_weight_deltas(network_prev_batch, network, weight_deltas)
        sum_error *= 1 / len(training_data) # Divide by lines of training data
        print(">Epoch %d \t|\t Error: %.3f" % (epoch, sum_error))

# Forward propagation function
def forward_prop(network, line, actv_func):
    # First input is from dataset
    inputs = line
    for layer in network:
        layer_output = [] # Will become input for next layer
        for neuron in layer:
            # Compute net input for neuron
            net_input = calc_net_input(neuron["w"], inputs)
            # Compute activation value of neuron
            neuron["b"] = activation_func(net_input, actv_func)
            layer_output.append(neuron["b"])
        inputs = layer_output
    return inputs

# Backward error propagation function
def back_prop_error(network, target, actv_func):
    # Start at last layer (output) and work backwards
    for i in range(len(network) - 1, -1, -1):
        layer = network[i]
        errors = list()
        if i == len(network) - 1: # For output layer:
            # Calculate discrepancy between target output and resultant output for neurons in output layer
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron["b"] - target[j])
        else: # For hidden layers:
            # Calculate net error 
            for j in range(len(layer)):
                error = 0
                for neuron in network[i - 1]:
                    # Gradient of error wrt input of node
                    error += neuron["w"][j] * neuron["d"]
                errors.append(error)
        for j in range(len(layer)):
            # Calcuate error of each neuron in the hidden layers
            neuron = layer[j]
            new_d = errors[j] * activation_func_prime(neuron["b"], actv_func)
            neuron["d"] = new_d

# Function to calculate net input for a neuron
def calc_net_input(weights, inputs):
    net_input = 0
    for i in range(len(weights) - 1):
        net_input += weights[i] * inputs[i] 
    return net_input + weights[-1]

# Function to update weights of the network
def update_weights(network, line, learn_rate, momentum, weight_deltas):
    for i in range(len(network) - 1, -1, -1):
        # Disclude the classification of the input
        inputs = line[:-1]
        if i != 0: # Use activation values from previous layer as inputs if not the first hidden layer
            inputs = [neuron["b"] for neuron in network[i - 1]]
        # Counter variable for iterating through "neurons" in weight_deltas
        a = 0
        for neuron in network[i]:
            for j in range(len(inputs)):
                # Adjust weights
                prev_delta = weight_deltas[i][a][j]
                new_delta = learn_rate * inputs[j] * neuron["d"] + momentum * prev_delta
                neuron["w"][j] -= new_delta
            a += 1
            neuron["w"][-1] -= learn_rate * neuron["d"]

# Function to update the weight deltas
def update_weight_deltas(network_prev_batch, network, weight_deltas):
    for i in range(len(network)):
        for j in range(len(network[i])):
            for k in range(len(network[i][j]["w"]) - 1):
                weight_deltas[i][j][k] = network[i][j]["w"][k] - network_prev_batch[i][j]["w"][k]
    return weight_deltas

# Activation function
def activation_func(input, function):
    if(function == 0): # Use Sigmoid
        if(input >= 0):
            return 1 / (1 + exp(-input))
        else:
            return 1 - (1 / (1 + exp(input)))
    else: # Use tanh
            return (exp(input) - exp(-input))/(exp(input) + exp(-input))

# Derivative of activation function
def activation_func_prime(output, function):
    if(function == 0):
        return output * (1 - output)
    else:
        return 1 - (activation_func(output, function) ** 2)

# Function to make predictions
def make_predictions(network, dataset, actv_func):
    predictions = list()
    for line in dataset:
        outputs = forward_prop(network, line, actv_func)
        predictions.append(outputs.index(max(outputs)))
    return predictions

# Function to calculate accuracy of predictions
def calc_accuracy(actual_output, predicted_output):
    correct_predictions = 0
    for i in range(len(actual_output)):
        if actual_output[i] == predicted_output[i]:
            correct_predictions += 1
    return correct_predictions / float(len(actual_output)) * 100

training_dataset = list()
testing_dataset = list()
# Read in data from files
for i in range(10):
    train_filename = "a1digits/digit_train_%d.txt" % i
    test_filename = "a1digits/digit_test_%d.txt" % i
    training_dataset += load_data_from_file(train_filename, i)
    testing_dataset += load_data_from_file(test_filename, i)
# Parse list elements to float
training_dataset = convert_list_to_floats(training_dataset)
testing_dataset = convert_list_to_floats(testing_dataset)

# Set parameters
batch_size = 32
learn_rate = 0.1
momentum = 0.2
num_epochs = 250
num_hidden_neurons = 4
actv_func = 0
validation_method = 0
k_folds = 5
validation_split = 0.3
# Get parameters from user
# batch_size = int(input("Size of batches: "))
# learn_rate = float(input("Learning rate: "))
# momentum = float(input("Momentum: "))
# num_epochs = int(input("Number of epochs: "))
# num_hidden_neurons = int(input("Number of neurons in hidden layer: "))
# actv_func = int(input("Activation function to use (0 = logistic, 1 = tanh): "))
# test_data_split = float(input("Enter the percentage of data to hold out for testing (as a decimal): "))
# k_folds = int(input("Number of folds (k): "))

results = list()
if(validation_method == 0): # Holdout validation
    training_data, validation_data = holdout_training_data(training_dataset, validation_split)
    predicted_output, best_model = do_neural_net(training_data, validation_data, batch_size, learn_rate, momentum, num_epochs, num_hidden_neurons, actv_func)
    actual_output = [line[-1] for line in validation_data]
    results.append(calc_accuracy(actual_output, predicted_output))
elif(validation_method == 1): # K-fold cross validation
    # Split dataset using k-fold cross validation
    k_split_dataset = crossval_split(training_dataset, k_folds)
    best_model_accuracy = 0
    best_model = None
    # Train k models, the kth model using the corresponding fold as it's validation data
    for fold in k_split_dataset:
        print("\nModel #", k_split_dataset.index(fold) + 1)
        training_data = list(k_split_dataset)
        validation_data = list()
        training_data.remove(fold)
        # k_split_database is a tuple of the folds and the data they contain - sum with empty list to reduce dimensionality
        training_data = sum(training_data, [])
        for line in fold:
            validation_data.append(line)
        predicted_output, model = do_neural_net(training_data, validation_data, batch_size, learn_rate, momentum, num_epochs, num_hidden_neurons, actv_func)
        actual_output = [line[-1] for line in validation_data]
        model_accuracy = calc_accuracy(actual_output, predicted_output)
        if(model_accuracy > best_model_accuracy):
            best_model = model
        results.append(model_accuracy)

print(best_model)
print("\nAverage accuracy on training/validation data: %0.2f" % (sum(results)/len(results)))
predictions = make_predictions(best_model, testing_dataset, actv_func)
actual_output = [line[-1] for line in testing_dataset]
model_accuracy = calc_accuracy(actual_output, predictions)
print("Accuracy of best model on testing data: %0.2f" % model_accuracy)
print("Batch size: ", batch_size,
    "\nLearn rate: ", learn_rate, 
    "\nMomentum: ", momentum, 
    "\nNum epochs: ", num_epochs, 
    "\nNum hidden neurons: ", num_hidden_neurons, 
    "\nActivation func: ", actv_func, 
    "\nK folds: ", k_folds)