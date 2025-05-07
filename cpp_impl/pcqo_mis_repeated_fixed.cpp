#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <string>

#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>
#include <limits.h>


// SATLIB settings
// #define LEARNING_RATE 0.0003
// #define MOMENTUM 0.875
// #define NUM_ITERATIONS 7500
// #define NUM_ITERATIONS_PER_BATCH 30
// #define GAMMA 900
// #define GAMMA_PRIME 1
// #define BATCH_SIZE 256
// #define STD 2.25

// ER settings
// #define LEARNING_RATE 0.000009
// #define MOMENTUM 0.9
// #define NUM_ITERATIONS 225000
// #define NUM_ITERATIONS_PER_BATCH 450
// #define GAMMA 350
// #define GAMMA_PRIME 7
// #define BATCH_SIZE 256
// #define STD 2.25

// Declare settings as global variables
// Read in the first argument as the file path
std::string FILE_PATH;
float LEARNING_RATE;
float MOMENTUM;
int NUM_ITERATIONS;
int NUM_ITERATIONS_PER_BATCH;
int GAMMA;
int GAMMA_PRIME;
int BATCH_SIZE;
float STD;
int OUTPUT_INTERVAL;
std::string INITIALIZATION_VECTOR;
int tuning_num_iter;
// Function to parse user input and set the global variables

torch::TensorOptions default_tensor_options = torch::TensorOptions().dtype(torch::kFloat16);
torch::TensorOptions default_tensor_options_gpu = default_tensor_options.device(torch::kCUDA);

class InitializationSampler
{
private:
    torch::Tensor mean_vector;
    float stdev;

public:
    InitializationSampler(torch::Tensor mean_vector, float stdev)
    {
        this->mean_vector = mean_vector;
        this->stdev = stdev;
    }
    torch::Tensor sample(torch::Tensor matrix)
    {
        auto std_tensor = torch::full_like(mean_vector, stdev);
        //torch::Tensor sample = torch::normal_out(matrix, mean_vector, STD, std::nullopt);
        torch::Tensor sample = torch::normal_out(matrix, mean_vector, std_tensor);
        return sample;
    }
    torch::Tensor sample_previous(torch::Tensor matrix)
    {
        mean_vector = matrix.clone();
        auto std_tensor = torch::full_like(mean_vector, stdev);
        //torch::Tensor sample = torch::normal_out(matrix, mean_vector, STD, std::nullopt);
        torch::Tensor sample = torch::normal_out(matrix, mean_vector, std_tensor);
        return sample;
    }
};

class Optimizer
{
public:
    float gamma;
    float gamma_prime;
    float learning_rate;
    float momentum;
    torch::Tensor velocity;
    Optimizer(float learning_rate, float momentum, float gamma, float gamma_prime, int graph_order, int batch_size)
    {
        this->learning_rate = learning_rate;
        this->momentum = momentum;
        this->velocity = torch::zeros({batch_size, graph_order}, default_tensor_options_gpu);
        this->gamma = gamma;
        this->gamma_prime = gamma_prime;
    }
    torch::Tensor compute_gradient(torch::Tensor &adjacency_matrix, torch::Tensor &adjacency_matrix_comp, torch::Tensor &X)
    {
        torch::Tensor first = torch::matmul(X, adjacency_matrix);
        torch::Tensor second = torch::matmul(X, adjacency_matrix_comp);
        torch::Tensor gradient = -1 + first.mul(gamma) - second.mul(gamma_prime);
        return gradient;
    }
    torch::Tensor velocity_update(torch::Tensor &X, torch::Tensor &gradient)
    {
        torch::Tensor momentum_term = velocity.mul(momentum);
        torch::Tensor learning_rate_term = gradient.mul(learning_rate);
        this->velocity = momentum_term.add(learning_rate_term);
        return X.sub(this->velocity);
    }
};

// Write a function that reads a graph from a file in DIMACS format and returns a cpp vector
torch::Tensor read_graph_from_file(const std::string &file_path)
{
    std::ifstream file(file_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file");
    }

    std::string header_line;

    while (std::getline(file, header_line))
    {
        if (header_line[0] == 'p')
        {
            break;
        }
    }

    // Header_line should look like this: "p edge 200 9876", we need to grab the second two numbers

    std::istringstream header_stream(header_line);
    std::string p, format;
    signed long number_of_nodes, number_of_edges;
    // std::cout << header_line << std::endl;
    header_stream >> p >> format >> number_of_nodes >> number_of_edges;
    // std::cout << "Number of nodes: " << number_of_nodes << std::endl;

    // torch::Tensor graph_entries = torch::zeros({number_of_nodes*number_of_nodes});
    std::vector<float> graph_entries(number_of_nodes * number_of_nodes, 0.0);

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream edge_stream(line);
        char a;
        size_t node_one, node_two;
        edge_stream >> a >> node_one >> node_two;
        graph_entries[(node_one - 1) * number_of_nodes + (node_two - 1)] = 1.0;
        graph_entries[(node_two - 1) * number_of_nodes + (node_one - 1)] = 1.0;
    }

    // Convert the vector to a tensor
    file.close();
    return torch::from_blob(graph_entries.data(), {number_of_nodes, number_of_nodes}).to(default_tensor_options_gpu);
}

torch::Tensor generate_complement_graph(const torch::Tensor &graph)
{
    return torch::ones({graph.sizes()[0], graph.sizes()[1]}, default_tensor_options_gpu) - graph - torch::eye(graph.sizes()[0], default_tensor_options_gpu);
}

std::tuple<torch::Tensor, torch::Tensor, long> load_graph(const std::string &file_path)
{
    torch::Tensor adjacency_matrix = read_graph_from_file(file_path);
    long number_of_nodes = adjacency_matrix.sizes()[0];
    torch::Tensor adjacency_matrix_comp = generate_complement_graph(adjacency_matrix);
    return std::make_tuple(adjacency_matrix, adjacency_matrix_comp, number_of_nodes);
}

torch::Tensor compute_mean_vector(const torch::Tensor &adjacency_matrix, const std::string &initialization_vector, int batch_size, int number_of_nodes)
{
    // Compute the degree of each node in the graph
    torch::Tensor degrees = adjacency_matrix.sum(0);

    // Compute the max degree of the graph
    float max_degree = degrees.max().item<float>();

    // Check if the initialization vector is provided and not empty
    torch::Tensor mean_vector;
    if (!initialization_vector.empty())
    {
        std::istringstream iss(initialization_vector);
        std::vector<float> mean_vector_data;
        std::string value;

        while (std::getline(iss, value, ' '))
        {
            mean_vector_data.push_back((value == "1") ? 1.0f : 0.0f);
        }

        if (mean_vector_data.size() != number_of_nodes)
        {
            throw std::runtime_error("Initialization vector size does not match the number of nodes in the graph");
        }

        mean_vector = torch::from_blob(mean_vector_data.data(), {number_of_nodes}, default_tensor_options).clone();
        mean_vector = mean_vector.unsqueeze(0).expand({batch_size, -1}).to(torch::kCUDA);
    }
    else
    {
        // Create the mean vector if no initialization vector is provided
        mean_vector = torch::zeros({number_of_nodes}, default_tensor_options);
        for (int i = 0; i < number_of_nodes; i++)
        {
            mean_vector[i] = 1.0 - (degrees[i] / (max_degree));
        }
        mean_vector = mean_vector.unsqueeze(0).expand({batch_size, -1}).to(torch::kCUDA);
    }

    return mean_vector;
}

struct Parameters
{
    float learning_rate;
    float momentum;
    int num_iterations_per_batch;
    int gamma;
    int gamma_prime;
    int batch_size;
    float std;
};

std::pair<std::vector<Parameters>, int> FindParameterRange(const std::string& file_path, int tuning_num_iter=1000)
{
    std::vector<Parameters> best_params_list;
    float best_score = 0;

    std::vector<float> learning_rates = {0.5, 0.05, 0.005, 0.0005, 0.0001, 0.00001, 0.000001, 0.0000001};
    //std::vector<float> learning_rates = {0.0000001, 0.000001, 0.00001, 0.0001, 0.0005, 0.005, 0.05, 0.5};
    //std::vector<float> learning_rates ={0.05, 0.005, 0.0005, 0.0001, 0.00001, 0.00009, 0.000001, 0.000009, 0.0000001};
    std::vector<float> momentums = {0.99, 0.9, 0.7};
    std::vector<int> gammas = {250, 500, 1000};
    std::vector<int> gamma_primes = {3, 5, 7};
    int num_iterations_per_batch = 500;
    int num_iterations = tuning_num_iter;
    int batch_size = 256;
    float std = 2.25;
    const int NUM_REPEATS = 5;

    torch::manual_seed(113);
    auto [adjacency_matrix, adjacency_matrix_comp, number_of_nodes] = load_graph(file_path);
    torch::Tensor mean_vector = compute_mean_vector(adjacency_matrix, INITIALIZATION_VECTOR, batch_size, number_of_nodes);
    InitializationSampler sampler = InitializationSampler(mean_vector, std);

    // fixed NUM_REAPEATS Xs
    std::vector<torch::Tensor> X_samples;
    for (int i = 0; i < NUM_REPEATS; i++) {
        torch::manual_seed(113 + i);
        X_samples.push_back(sampler.sample(torch::zeros({batch_size, number_of_nodes}, default_tensor_options_gpu)));
    }

    int total_iterations = learning_rates.size() * momentums.size() * gammas.size() * gamma_primes.size();
    int current_iteration = 0;
    int total_run = 0;

    while (best_params_list.empty())
    {
        for (float lr : learning_rates)
        for (float mom : momentums)
        for (int gamma : gammas)
        for (int gamma_prime : gamma_primes)
        {
            current_iteration++;
            std::cout << "optimizing: " << current_iteration << "/" << total_iterations << std::endl;

            float cumulative_score = 0;

            for (int r = 0; r < NUM_REPEATS; ++r)
            {
                // Copy Xs
                torch::Tensor X = X_samples[r].clone();
                Optimizer optimizer = Optimizer(lr, mom, gamma, gamma_prime, number_of_nodes, batch_size);

                int local_max = 0;
                torch::Tensor ones_vector = torch::ones({number_of_nodes}, default_tensor_options_gpu);
                torch::Tensor update = number_of_nodes * adjacency_matrix - adjacency_matrix_comp;

                for (int iteration = 0; iteration < num_iterations; iteration++)
                {
                    torch::Tensor gradient = optimizer.compute_gradient(adjacency_matrix, adjacency_matrix_comp, X);
                    X = optimizer.velocity_update(X, gradient);
                    X = X.clamp(0, 1);

                    if ((iteration + 1) % num_iterations_per_batch == 0)
                    {
                        torch::Tensor masks = X.gt(0.5).to(torch::kFloat16);

                        for (int i = 0; i < masks.sizes()[0]; i++)
                        {
                            torch::Tensor binarized_update = masks[i] - 0.1 * (-ones_vector + masks[i].matmul(update));
                            binarized_update.clamp_(0, 1);
                            if (torch::equal(binarized_update, masks[i]))
                            {
                                int current_score = masks[i].sum().item<float>();
                                if (current_score > local_max)
                                    local_max = current_score;
                            }
                        }
                        X = sampler.sample_previous(X);
                    }
                }

                cumulative_score += local_max;
            }

            float avg_score = cumulative_score / NUM_REPEATS;

            if (avg_score > best_score)
            {
                best_score = avg_score;
                best_params_list.clear();
                //best_params_list.push_back({lr, mom, num_iterations_per_batch, gamma, gamma_prime, batch_size, std});
            }
            if (best_score != 0 && avg_score == best_score)
            {
                std::cout << "max == best_score: " << best_score << std::endl;
                best_params_list.push_back({lr, mom, num_iterations_per_batch, gamma, gamma_prime, batch_size, std});
                std::cout << "best hyperparameters:\n";
                for (const auto& param : best_params_list) {
                    std::cout << param.learning_rate << ", " << param.momentum  << ", " << param.gamma  << ", " << param.gamma_prime << std::endl; 
                }
                
            }
        }

        total_run += num_iterations * total_iterations;

        if (best_params_list.size() == 0)
        {
            // current_iteration = 0;
            // num_iterations_per_batch = num_iterations_per_batch * 2;
            // num_iterations = num_iterations * 2;
            // gammas.push_back(gammas[gammas.size() - 1] * 2);
            // std::cout << "No parameters found, increasing gamma and num_iterations_per_batch" << std::endl;
            // std::cout << "Increasing gamma to: " << gammas[gammas.size() - 1] << std::endl;
            // std::cout << "Increasing num_iterations_per_batch to: " << num_iterations_per_batch << std::endl;
            std::cout << "No parameters found" << std::endl;
            break;
        }
    }
    for (const auto& param : best_params_list) {
        std::cout << param.learning_rate << ", " << param.momentum  << ", " << param.gamma  << ", " << param.gamma_prime << std::endl; 
    }
    return {best_params_list, total_run};
}


bool directoryExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool createDirectories(const std::string& path) {
    char temp[PATH_MAX];
    std::strncpy(temp, path.c_str(), sizeof(temp));
    temp[sizeof(temp) - 1] = '\0';

    for (char* p = temp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (!directoryExists(temp)) {
                if (mkdir(temp, 0755) != 0 && errno != EEXIST) {
                    perror("mkdir");
                    return false;
                }
            }
            *p = '/';
        }
    }

    if (!directoryExists(temp)) {
        if (mkdir(temp, 0755) != 0 && errno != EEXIST) {
            perror("mkdir");
            return false;
        }
    }

    return true;
}


std::string convertPathToCSV(const std::string& file_path, int tuning_iter) {
    std::string new_path = file_path;

    // "graphs/" -> "fast_tune/"
    size_t graphs_pos = new_path.find("/graphs/");
    if (graphs_pos != std::string::npos) {
        new_path.replace(graphs_pos, 8, "/fast_tune/");
    }
    // ".clq" -> "_{tuning_iter}.csv"
    size_t ext_pos = new_path.rfind(".clq");
    if (ext_pos != std::string::npos) {
        std::string suffix = "_tuningiter" + std::to_string(tuning_iter) + "_repeated_fixed_sample.csv";
        new_path.replace(ext_pos, 4, suffix);
    }

    //create directory
    size_t last_slash = new_path.rfind('/');
    if (last_slash != std::string::npos) {
        std::string dir_path = new_path.substr(0, last_slash);
        createDirectories(dir_path); 
    }

    return new_path;
}
void parse_user_input(int argc, const char *argv[])
{
    if (argc > 10)
    {
        FILE_PATH = argv[1];
        LEARNING_RATE = std::stof(argv[2]);
        MOMENTUM = std::stof(argv[3]);
        NUM_ITERATIONS = std::stoi(argv[4]);
        NUM_ITERATIONS_PER_BATCH = std::stoi(argv[5]);
        GAMMA = std::stoi(argv[6]);
        GAMMA_PRIME = std::stoi(argv[7]);
        BATCH_SIZE = std::stoi(argv[8]);
        STD = std::stof(argv[9]);
        OUTPUT_INTERVAL = std::stoi(argv[10]);
    }
    else
    {
        if (argc > 1)
        {
            FILE_PATH = argv[1];
            tuning_num_iter = std::stoi(argv[2]);
            std::cout << "Not enough parameters provided, performing grid search to find appropriate parameters" << std::endl;
            auto start_tune = std::chrono::high_resolution_clock::now();
            //std::vector<Parameters> parameters = FindParameterRange(FILE_PATH);
            auto result = FindParameterRange(FILE_PATH, tuning_num_iter);
            auto end_tune = std::chrono::high_resolution_clock::now();
            std::vector<Parameters> parameters = result.first;
                for (const auto& param : parameters) {
                    std::cout << param.learning_rate << ", " << param.momentum  << ", " << param.gamma  << ", " << param.gamma_prime << std::endl; 
                }
            int total_run = result.second;
            
            std::chrono::duration<double> tuning_seconds = end_tune - start_tune;
            std::cout << "tuning time: " << tuning_seconds.count() << " sec" << std::endl;
            if (!parameters.empty())
            {
                float min_learning_rate = parameters[0].learning_rate, max_learning_rate = parameters[0].learning_rate;
                float min_momentum = parameters[0].momentum, max_momentum = parameters[0].momentum;
                int min_gamma = parameters[0].gamma, max_gamma = parameters[0].gamma;
                int min_gamma_prime = parameters[0].gamma_prime, max_gamma_prime = parameters[0].gamma_prime;
                int min_batch_size = parameters[0].batch_size, max_batch_size = parameters[0].batch_size;
                float min_std = parameters[0].std, max_std = parameters[0].std;
                int min_num_iterations_per_batch = parameters[0].num_iterations_per_batch, max_num_iterations_per_batch = parameters[0].num_iterations_per_batch;

                for (const auto &param : parameters)
                {
                    min_learning_rate = std::min(min_learning_rate, param.learning_rate);
                    max_learning_rate = std::max(max_learning_rate, param.learning_rate);
                    min_momentum = std::min(min_momentum, param.momentum);
                    max_momentum = std::max(max_momentum, param.momentum);
                    min_gamma = std::min(min_gamma, param.gamma);
                    max_gamma = std::max(max_gamma, param.gamma);
                    min_gamma_prime = std::min(min_gamma_prime, param.gamma_prime);
                    max_gamma_prime = std::max(max_gamma_prime, param.gamma_prime);
                    min_batch_size = std::min(min_batch_size, param.batch_size);
                    max_batch_size = std::max(max_batch_size, param.batch_size);
                    min_std = std::min(min_std, param.std);
                    max_std = std::max(max_std, param.std);
                    min_num_iterations_per_batch = std::min(min_num_iterations_per_batch, param.num_iterations_per_batch);
                    max_num_iterations_per_batch = std::max(max_num_iterations_per_batch, param.num_iterations_per_batch);
                }

                std::cout << "Parameter ranges found:" << std::endl;
                std::cout << "Learning rate: [" << min_learning_rate << ", " << max_learning_rate << "]" << std::endl;
                std::cout << "Momentum: [" << min_momentum << ", " << max_momentum << "]" << std::endl;
                std::cout << "Gamma: [" << min_gamma << ", " << max_gamma << "]" << std::endl;
                std::cout << "Gamma prime: [" << min_gamma_prime << ", " << max_gamma_prime << "]" << std::endl;
                std::cout << "Batch size: [" << min_batch_size << ", " << max_batch_size << "]" << std::endl;
                std::cout << "Std: [" << min_std << ", " << max_std << "]" << std::endl;
                std::cout << "Num iterations per batch: [" << min_num_iterations_per_batch << ", " << max_num_iterations_per_batch << "]" << std::endl;

                // Pick a set of parameters in the middle of the list
                size_t middle_index = parameters.size() / 2;
                const auto &selected_param = parameters[middle_index];

                // Set the global variables to the selected parameters
                LEARNING_RATE = selected_param.learning_rate;
                MOMENTUM = selected_param.momentum;
                NUM_ITERATIONS_PER_BATCH = selected_param.num_iterations_per_batch;
                GAMMA = selected_param.gamma;
                GAMMA_PRIME = selected_param.gamma_prime;
                BATCH_SIZE = selected_param.batch_size;
                STD = selected_param.std;
                NUM_ITERATIONS = NUM_ITERATIONS_PER_BATCH * 10;
                OUTPUT_INTERVAL = 1;

                std::cout << "Selected parameters:" << std::endl;
                std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
                std::cout << "Momentum: " << MOMENTUM << std::endl;
                std::cout << "Gamma: " << GAMMA << std::endl;
                std::cout << "Gamma prime: " << GAMMA_PRIME << std::endl;
                std::cout << "Batch size: " << BATCH_SIZE << std::endl;
                std::cout << "Std: " << STD << std::endl;
                std::cout << "Num iterations per batch: " << NUM_ITERATIONS_PER_BATCH << std::endl;
                std::cout << "Num iterations: " << NUM_ITERATIONS << std::endl;

                std::string csv_filename = convertPathToCSV(FILE_PATH, tuning_num_iter);
                
                std::ofstream csvFile(csv_filename);
                if (!csvFile.is_open()) {
                    std::cerr << "Unable to open CSV: " << csv_filename << "\n";
                    exit(1);
                }
                csvFile << "LearningRate,Momentum,Gamma,GammaPrime,BatchSize,Std,NumIterPerBatch,NumIterations,TuningTime,TuningRun\n";
                csvFile << LEARNING_RATE << ","
                        << MOMENTUM << ","
                        << GAMMA << ","
                        << GAMMA_PRIME << ","
                        << BATCH_SIZE << ","
                        << STD << ","
                        << NUM_ITERATIONS_PER_BATCH << ","
                        << NUM_ITERATIONS << ","
                        << tuning_seconds.count() << ","
                        << total_run << "\n";

                csvFile.close();
                std::cout << "CSV Saved: " << csv_filename << "\n";
            }
            else
            {
                std::cout << "No parameters found." << std::endl;
                exit(1);
            }
        }
        else
        {
            std::cerr << "Usage: " << argv[0] << " <file_path> [learning_rate] [momentum] [num_iterations] [num_iterations_per_batch] [gamma] [gamma_prime] [batch_size] [std] [output_interval] [initialization_vector]" << std::endl;
            exit(1);
        }
    }
    if (argc > 11)
    {
        INITIALIZATION_VECTOR = argv[11];
    }
}

int main(int argc, const char *argv[])
{
    int sum_max = 0;
    int count = 0;
    
    parse_user_input(argc, argv);

    std::cout << FILE_PATH << "-" << GAMMA << "-" << LEARNING_RATE << "-" << GAMMA_PRIME << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    torch::manual_seed(113);

    auto [adjacency_matrix, adjacency_matrix_comp, number_of_nodes] = load_graph(FILE_PATH);

    // compute mean vector
    torch::Tensor mean_vector = compute_mean_vector(adjacency_matrix, INITIALIZATION_VECTOR, BATCH_SIZE, number_of_nodes);

    InitializationSampler sampler = InitializationSampler(mean_vector, STD);

    // Create initilzation matrix and sample from a normal distribution with mean_vector and std 0.01
    torch::Tensor X = sampler.sample(torch::zeros({BATCH_SIZE, number_of_nodes}, default_tensor_options_gpu));

    // //std::cout << "Initialization matrix: " << X << std::endl;

    Optimizer optimizer = Optimizer(LEARNING_RATE, MOMENTUM, GAMMA, GAMMA_PRIME, number_of_nodes, BATCH_SIZE);

    int max = 0;
    torch::Tensor max_vector;
    // std::cout << "Starting optimization" << std::endl;

    torch::Tensor ones_vector = torch::ones({number_of_nodes}, default_tensor_options_gpu);
    torch::Tensor update = number_of_nodes * adjacency_matrix - adjacency_matrix_comp;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++)
    {
        torch::Tensor gradient = optimizer.compute_gradient(adjacency_matrix, adjacency_matrix_comp, X);
        X = optimizer.velocity_update(X, gradient);

        // Clamp the initialization matrix to be between 0 and 1
        X = X.clamp(0, 1);

        if ((iteration + 1) % NUM_ITERATIONS_PER_BATCH == 0)
        {
            torch::Tensor masks = X.gt(0.5).to(torch::kFloat16);
            // Iterate over the batch dimension of the masks tensor
            // torch::Tensor sums = masks.sum(1);

            for (int i = 0; i < masks.sizes()[0]; i++)
            {
                // if (sums[i].item<float>() > 0 && masks[i].t().matmul(adjacency_matrix).matmul(masks[i]).item<float>() == 0)
                // {
                torch::Tensor binarized_update = masks[i] - 0.1 * (-ones_vector + masks[i].matmul(update));
                binarized_update.clamp_(0, 1);
                if (torch::equal(binarized_update, masks[i]))
                {
                    if (masks[i].sum().item<float>() > max)
                    {
                        max = masks[i].sum().item<float>();
                        max_vector = masks[i].clone();
                    }
                }
                // }
            }

            if (iteration + 1 == NUM_ITERATIONS_PER_BATCH || ((iteration + 1) / NUM_ITERATIONS_PER_BATCH) % OUTPUT_INTERVAL == 0 || iteration + 1 == NUM_ITERATIONS)
            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                std::cout << (iteration + 1) / NUM_ITERATIONS_PER_BATCH << std::endl;
                std::cout << max << std::endl;
                std::cout << elapsed_seconds.count() << std::endl;
            }
            if (iteration + 1 != NUM_ITERATIONS)
            {
                X = sampler.sample_previous(X);
            }
            else
            {
                // Print the max vector on one line
                for (int i = 0; i < max_vector.sizes()[0]; i++)
                {
                    std::cout << max_vector[i].item<float>() << (i == max_vector.sizes()[0] - 1 ? "\n" : " ");
                }
            }
        }
    }
}