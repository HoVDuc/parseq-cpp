#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include "tokenizer.h"
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <numeric>

std::string charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
Tokenizer tokenizer(charset);

const std::string PATH = "/home/anlab/hovduc/parseq-trace/parseq/test/0418/";
torch::jit::script::Module module = torch::jit::load("/home/anlab/hovduc/parseq-trace/parseq.pt");

at::Tensor preprocess(cv::Mat& image) {
    cv::resize(image, image, cv::Size(128, 32), cv::INTER_CUBIC);

    CV_Assert(image.rows == 32 && image.cols == 128);
    CV_Assert(image.type() == CV_8UC3);

    // Convert the image from BGR to RGB and to float type
    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0 / 255);

    // Create a tensor from the image
    at::Tensor im_tensor = torch::from_blob(image_rgb.data, {1, 32, 128, 3}, torch::kFloat);

    // Permute the dimensions to match the shape (1, 3, 32, 128)
    im_tensor = im_tensor.permute({0, 3, 1, 2});

    // Normalize the tensor
    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> std_dev = {0.229, 0.224, 0.225};
    im_tensor = torch::data::transforms::Normalize<>(mean, std_dev)(im_tensor);

    return im_tensor.clone(); // Clone if the original tensor's data should be detached from the OpenCV Mat
}

std::vector<std::string> get_data(std::string filename)
{
    std::ifstream gtruthData(filename);
    std::vector<std::string> data;
    std::string line;

    if (gtruthData.is_open())
    {
        while (getline(gtruthData, line))
        {
            data.push_back(line);
        }
        gtruthData.close();
    }
    else
    {
        std::cerr << "Unable to open file" << std::endl;
    }
    return data;
}

int main(int argc, const char *argv[])
{
    int BATCH_SIZE = 1;
    const std::string filename = "/home/anlab/hovduc/parseq-trace/parseq/test/0418/test_gt.txt";
    std::vector<std::string> pathList;
    std::vector<std::string> gtList;
    std::vector<std::string> predList;


    std::vector<std::string> dataset = get_data(filename);
    for (const auto &line : dataset)
    {
        std::istringstream iss(line);
        std::string path, gt;
        if (std::getline(iss, path, '\t') && std::getline(iss, gt, '\t'))
        {
            pathList.push_back(path);
            gtList.push_back(gt);
        }
    }
    int num_batch = (pathList.size() + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int i = 0; i < num_batch; i++)
    {
        std::vector<at::Tensor> images;
        int batch_start = i * BATCH_SIZE;
        int batch_end = std::min((i + 1) * BATCH_SIZE, (int)pathList.size());

        for (int j = batch_start; j < batch_end; ++j)
        {
            cv::Mat image = cv::imread(PATH + pathList[j]);
            at::Tensor tensor = preprocess(image);

            // Create a vector of IValues and add the tensor to the vector
            images.push_back(tensor);
        }
        auto images_concat = torch::cat(images, 0);
        at::Tensor outputs = module.forward({images_concat}).toTensor();
        auto detached_token_dists = outputs.detach();
        
        // Zero out specific token distributions
        detached_token_dists = detached_token_dists.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(11, 74)}, torch::zeros({detached_token_dists.size(0), detached_token_dists.size(1), 63}));
        detached_token_dists = detached_token_dists.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(75, 76)}, torch::zeros({detached_token_dists.size(0), detached_token_dists.size(1), 1}));
        detached_token_dists = detached_token_dists.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(77, torch::indexing::None)}, torch::zeros({detached_token_dists.size(0), detached_token_dists.size(1), detached_token_dists.size(2) - 77}));

        // Predict tokens
        auto [tokens, probs] = tokenizer.decode(detached_token_dists, false, true);
        std::replace(tokens.begin(), tokens.end(), ",", ".");

        predList.push_back(tokens[0]);
    }

    std::ofstream outfile;
    outfile.open("../predict.txt", std::ios_base::app);
    for (int i = 0; i < pathList.size(); i++)
    {
        std::string text = pathList[i] + "\t" + gtList[i] + "\t" + predList[i] + "\n";
        outfile << text;
    }
    return 0;
}
