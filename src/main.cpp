#include <iostream>

#include "transformer.hpp"

using transformer::make_transformers;
using transformer::make_lazy_transformer;
using transformer::Binarizer;

struct Data {
    std::string firstname;
    std::string lastname;
};

int main(int argc, char *argv[])
{
    // Let's do
    std::function<std::string(const Data& sample)> get_firstname_lambda =
        [](const Data& sample) -> std::string {
            return sample.firstname;
        };
    std::function<std::string(const Data& sample)> get_lastname_lambda =
        [](const Data& sample) -> std::string {
            return sample.lastname;
        };

    auto get_firstname = make_lazy_transformer(get_firstname_lambda);
    auto get_lastname = make_lazy_transformer(get_lastname_lambda);
    auto binarizer = make_transformers<Binarizer<std::string>>();

    Data data1{"Mike", "Jordan"};
    Data data2{"Mike", "James"};
    Data data3{"Bill", "Jordan"};
    Data data4{"Bill", "James"};

    auto pipe = (get_firstname | get_lastname) + binarizer;
    auto dataset = {data1, data2, data3, data4};
    for (const auto& data: dataset) {
        pipe->step(data);
    }
    pipe->finalize();
    for (const auto& data: dataset) {
        std::vector<double> out = pipe->transform(data);
        for (const auto& item: out) {
            std::cout<<item<<" ";
        }
        std::cout<<std::endl;
    }
    // Output will be:
    // 1 0 0 0
    // 0 1 0 0
    // 0 0 1 0
    // 0 0 0 1
    return 0;
}
