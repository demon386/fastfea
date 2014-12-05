/**
 * Transformer is extended from the idea from scikit-learn for feature
 * engineering.
 *
 * Transformer is a kind of "micro-model", it fits the whole dataset to
 * determine its paramters. After that, it is not used to predict label, but
 * transform features based on its paramters.
 *
 * Some examples of transformer:
 *
 * - Standardizer: Standardize a feature by its global mean and variance across
 * all samples. In order to do that, obviously, it has to fit all samples to get
 * mean and variance.
 * - Binarizer for categorical (discrete) feature: it has to go through all
 * samples to see how many levels a partuclar categorical variable has.
 */
#ifndef FASTFEA_TRANSFORMER_H
#define FASTFEA_TRANSFORMER_H

#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <memory>

#include "hasher.hpp"

namespace transformer {

template<class From, class To>
using TransformFunc = std::function<To(const From& sampl)>;

template<typename From, typename To>
class Transformer {
public:
    using BaseType = Transformer<From, To>;

    virtual ~Transformer() {}
    /**
     * Step will be called when going through each sample.
     *
     * Since multiple transformers can be applied to a dataset, this is more
     * efficient than letting each transformer getting the whole dataset and
     * going through.
     */
    virtual void step(const From& sample) {}
    virtual void step(From&& sample) {
        step(sample);
    }
    /**
     * After finishing all samples, this function will be called.
     */
    virtual void finalize() {};
    /**
     * Transform new data, return a vector of string.
     * Here we use double because it's general, and fastfea is not responsible
     * for training, but rather output a numeric feature set for other modeling
     * to train.
     */
    virtual To transform(const From& sample) const = 0;
    virtual To transform(From&& sample) {
        return transform(sample);
    }
    bool is_finalized() const {
        return _is_finalized;
    }

protected:
    bool _is_finalized = true;
};

/**
 * 1-of-K coding
 * e.g. 0001, 0010, 0100, 1000 for 4-level categorical variable.
 */
template<typename From>
class Binarizer : public Transformer<From, std::vector<double>> {
public:
    Binarizer() { this->_is_finalized = false; }
    virtual void step(const From& sample) {
        if (_data_to_val.find(sample) == _data_to_val.end()) {
            _data_to_val[sample] = _count++;
        }
    }

    virtual std::vector<double> transform(const From& sample) const {
        int val = _data_to_val.at(sample);
        std::vector<double> output;
        for (int i = 0; i < _count; i++) {
            if (val == i) {
                output.emplace_back(1.0);
            } else {
                output.emplace_back(0.0);
            }
        }
        return output;
    }

private:
    int _count = 0;
    std::unordered_map<From, int> _data_to_val;
};

template<typename From, typename Middle, typename To>
class Pipeline : public Transformer<From, To> {
public:
    Pipeline(const std::shared_ptr<Transformer<From, Middle>> first,
            const std::shared_ptr<Transformer<Middle, To>> second):
            _first(first), _second(second) {
        if (_first->is_finalized() && _second->is_finalized()) {
            this->_is_finalized = true;
        } else {
            this->_is_finalized = false;
        }
    }

    virtual void step(const From& sample) {
        if (this->is_finalized()) {
            return;
        }

        if (_first->is_finalized()) {
            if (!_second->is_finalized()) {
                _second->step(_first->transform(sample));
            }
        }
        else {
            _first->step(sample);
            if (!_second->is_finalized()) {
                _data.emplace(sample);
            }
        }
    }

    virtual void finalize() {
        if (!_first->is_finalized()) {
            _first->finalize();
        }
        if (!_second->is_finalized()) {
            while (!_data.empty()) {
                _second->step(_first->transform(_data.front()));
                _data.pop();
            }
            _second->finalize();
        }
        this->_is_finalized = true;
    }

    virtual To transform(const From& sample) const {
        return _second->transform(_first->transform(sample));
    }

private:
    std::shared_ptr<Transformer<From, Middle>> _first;
    std::shared_ptr<Transformer<Middle, To>> _second;
    std::queue<From> _data;
};


template<typename T1, typename T2>
std::tuple<T1, T2> combine(T1&& first_out, T2&& second_out) {
    return std::make_tuple(std::move(first_out), std::move(second_out));
}

template<class... Args, class T2>
auto combine(std::tuple<Args...>&& first, T2&& second) ->
        decltype(std::tuple_cat(first, std::make_tuple(second))) {
    return std::tuple_cat(std::move(first), std::make_tuple(std::move(second)));
}

template<class T1, class... Args>
auto combine(T1&& first, std::tuple<Args...>&& second) ->
        decltype(std::tuple_cat(std::make_tuple(first), second)) {
    return std::tuple_cat(std::make_tuple(std::move(first)),
            std::move(second));
}

template<class... Args1, class... Args2>
auto combine(std::tuple<Args1...>&& first,
        std::tuple<Args2...>&& second) ->
        decltype(std::tuple_cat(first, second)) {
    return std::tuple_cat(std::move(first),
            std::move(second));
}

template<typename T>
std::vector<T> combine(std::vector<T>&& first_out,
        std::vector<T>&& second_out) {
    std::vector<T> out(std::move(first_out));
    out.insert(out.end(), std::move(second_out).begin(),
        std::move(second_out).end());
    return out;
}

// Combiner, by itself, just call two transformer in sequence with the same
// input.
// It's useless as standalone, but can combine with pipeline to provide
// powerful abstraction.
template<typename From, typename To1, typename To2>
class Combiner : public Transformer<From, decltype(combine(To1(),
            To2()))> {
    using Transformer1T = Transformer<From, To1>;
    using Transformer2T = Transformer<From, To2>;
    using CombineT = decltype(combine(To1(), To2()));
    using TransformerCombineT = Transformer<From, CombineT>;
public:
    Combiner(const std::shared_ptr<Transformer1T> first,
            const std::shared_ptr<Transformer2T> second) :
            _first(std::move(first)), _second(std::move(second)) {
        this->_is_finalized = false;
    }

    virtual void step(const From& sample) {
        if (!_first->is_finalized()) {
            _first->step(sample);
        }
        if (!_second->is_finalized()) {
            _second->step(sample);
        }
    }

    virtual void finalize() {
        if (!_first->is_finalized()) {
            _first->finalize();
        }
        if (!_second->is_finalized()) {
            _second->finalize();
        }
        this->_is_finalized = true;
    }

    virtual CombineT transform(const From& sample) const {
        return combine(_first->transform(sample),
            _second->transform(sample));
    }

private:
    std::shared_ptr<Transformer1T> _first;
    std::shared_ptr<Transformer2T> _second;
};

template<typename From, typename Middle, typename To>
std::shared_ptr<Transformer<From, To>> operator+(
        std::shared_ptr<Transformer<From, Middle>> first,
        std::shared_ptr<Transformer<Middle, To>> second) {
    return std::shared_ptr<Transformer<From, To>>(new Pipeline<From, Middle, To>(first, second));
}

template<typename From, typename To1, typename To2>
std::shared_ptr<Transformer<From,
    decltype(combine(To1(), To2()))>> operator|(
        std::shared_ptr<Transformer<From, To1>> first,
        std::shared_ptr<Transformer<From, To2>> second) {
    using CombineT = decltype(combine(To1(), To2()));
    return std::shared_ptr<Transformer<From, CombineT>>(new Combiner<From, To1, To2>(first, second));
}

template<class T, class... Args>
std::shared_ptr<typename T::BaseType> make_transformer(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

// Lazy transformer doesn't need to fit anything.
//
// Analogy: In the world of classifiers, k-NN doesn't need to "fit" and it's a
// lazy model.
template<typename From, typename To>
class LazyTransformer : public Transformer<From, To> {
public:
    LazyTransformer(std::function<To(const From& sample)> func) : _func(func) {
        this->_is_finalized = true;
    }

    virtual To transform(const From& sample) const {
        return _func(sample);
    }
private:
    std::function<To(const From& sample)> _func;
};

template<class From, typename To>
std::shared_ptr<Transformer<From, To>> make_lazy_transformer(
        std::function<To(const From& sample)> func) {
    return std::make_shared<LazyTransformer<From, To>>(std::move(func));
}
} // namsepace: transformer

#endif
