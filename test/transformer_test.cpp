#include <gtest/gtest.h>
#include <string>

#include "transformer.hpp"

using transformer::Transformer;
using transformer::make_lazy_transformer;
using transformer::TransformFunc;

// Simple data for testing
struct Data {
    std::string firstname;
    std::string lastname;
};

auto firstname_lambda = [](const Data& sample) -> std::string {
    return sample.firstname;
};
auto lastname_lambda = [](const Data& sample) -> std::string {
    return sample.lastname;
};
// This is ugly but this is the only way I know to transform from lambda to
// std::function.
//
// See http://stackoverflow.com/questions/13358672/how-to-convert-a-lambda-to-an-stdfunction-using-templates
template<typename Lambda>
auto make_lazy_data_transformer(Lambda lambda)
        -> std::shared_ptr<
        Transformer<Data, decltype(lambda(Data()))>
        > {
    using FuncReturnType = decltype(lambda(Data()));
    using FuncType = std::function<FuncReturnType(const Data&)>;
    return make_lazy_transformer(static_cast<FuncType>(lambda));
}

TEST(transformer, lazy_transformer) {
    auto lambda = [](const Data& sample) -> int {
        return sample.firstname.length();
    };
    auto firstname_length = make_lazy_data_transformer(lambda);

    Data data{"Michael", "Jordan"};
    auto out = firstname_length->transform(data);
    EXPECT_EQ(7, out);
}

TEST(transformer, pipeline) {
    TransformFunc<std::string,int> length_lambda = [](const std::string& str) -> int {
        return str.length();
    };

    auto firstname = make_lazy_data_transformer(firstname_lambda);
    auto length = make_lazy_transformer(length_lambda);
    auto pipe = firstname + length;

    Data data{"Michael", "Jordan"};
    auto firstname_out = firstname->transform(data);
    EXPECT_EQ("Michael", firstname_out);
    auto out = pipe->transform(data);
    EXPECT_EQ(7, out);
}

TEST(transformer, combiner) {
    auto firstname = make_lazy_data_transformer(firstname_lambda);
    auto lastname = make_lazy_data_transformer(lastname_lambda);
    auto combiner = firstname | lastname;

    Data data{"Michael", "Jordan"};
    auto out = combiner->transform(data);
    EXPECT_EQ("Michael", std::get<0>(out));
    EXPECT_EQ("Jordan", std::get<1>(out));
}

TEST(combiner, three_and_four_combiners) {
    auto firstname = make_lazy_data_transformer(firstname_lambda);
    auto lastname = make_lazy_data_transformer(lastname_lambda);
    auto combiner = firstname | lastname | firstname;

    Data data{"Michael", "Jordan"};
    auto out = combiner->transform(data);
    EXPECT_EQ("Michael", std::get<0>(out));
    EXPECT_EQ("Jordan", std::get<1>(out));
    EXPECT_EQ("Michael", std::get<2>(out));

    combiner = firstname | (firstname | lastname);
    out = combiner->transform(data);
    EXPECT_EQ("Michael", std::get<0>(out));
    EXPECT_EQ("Michael", std::get<1>(out));
    EXPECT_EQ("Jordan", std::get<2>(out));

    auto combiner4 = (firstname | lastname) | (lastname | firstname);
    auto out4 = combiner4->transform(data);
    EXPECT_EQ("Michael", std::get<0>(out4));
    EXPECT_EQ("Jordan", std::get<1>(out4));
    EXPECT_EQ("Jordan", std::get<2>(out4));
    EXPECT_EQ("Michael", std::get<3>(out4));
}