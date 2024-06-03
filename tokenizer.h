#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <optional>
#include <stdexcept>
#include <iostream>
#include <algorithm>

class BaseTokenizer {
public:
    BaseTokenizer(const std::string& charset, const std::vector<char>& specials_first = {}, const std::vector<char>& specials_last = {}) {
        _itos.reserve(specials_first.size() + charset.size() + specials_last.size());
        _itos.insert(_itos.end(), specials_first.begin(), specials_first.end());
        _itos.insert(_itos.end(), charset.begin(), charset.end());
        _itos.insert(_itos.end(), specials_last.begin(), specials_last.end());

        for (size_t i = 0; i < _itos.size(); ++i) {
            _stoi[_itos[i]] = i;
        }
    }

    size_t size() const {
        return _itos.size();
    }

    std::vector<int> tok2ids(const std::string& tokens) const {
        std::vector<int> ids;
        ids.reserve(tokens.size());
        for (char s : tokens) {
            ids.push_back(_stoi.at(s));
        }
        return ids;
    }

    std::string ids2tok(const std::vector<int>& token_ids, bool join = true) const {
        std::string tokens;
        std::vector<int> sb_tokens(token_ids.begin(), token_ids.begin() + 6);
        tokens.reserve(sb_tokens.size());
        for (int id : sb_tokens) {
            if (id == 0) {
                break;
            }
            id = id == 74 ? 76 : id;
            tokens.push_back(_itos.at(id));
        }
        return join ? tokens : std::string{tokens.begin(), tokens.end()};
    }

    /* Encode
    virtual torch::Tensor encode(const std::vector<std::string>& labels, const std::optional<torch::Device>& device = std::nullopt) const = 0;
    */
    virtual std::tuple<torch::Tensor, std::vector<int>> myfilter(const torch::Tensor& probs, const torch::Tensor& ids) const = 0;

    std::tuple<std::vector<std::string>, std::vector<torch::Tensor>> decode(const torch::Tensor& token_dists, bool raw = false, bool read = false) const {
        std::vector<std::string> batch_tokens;
        std::vector<torch::Tensor> batch_probs;

        for (int i = 0; i < token_dists.size(0); ++i) {
            auto dist = token_dists[i];
            auto [probs, ids] = dist.max(-1);
            ids = ids.to(torch::kInt32);
        
            std::vector<int> id_list(ids.data_ptr<int>(), ids.data_ptr<int>() + ids.numel());
            std::string tokens = ids2tok(id_list, !raw);
            batch_tokens.push_back(tokens);

            auto mean_prob = probs.slice(0, 0, probs.size(0) - 1).mean().item<float>();
            batch_probs.push_back(torch::tensor(mean_prob));
        }

        return std::make_tuple(batch_tokens, batch_probs);
    }

protected:
    std::vector<char> _itos;
    std::unordered_map<char, int> _stoi;
};

class Tokenizer : public BaseTokenizer {
public:
    static constexpr char BOS = 'B';
    static constexpr char EOS = 'E';
    static constexpr char PAD = 'P';

    Tokenizer(const std::string& charset)
        : BaseTokenizer(charset, {EOS}, {BOS, PAD}) {
        eos_id = _stoi.at(EOS);
        bos_id = _stoi.at(BOS);
        pad_id = _stoi.at(PAD);
    }

    /* Encode
    torch::Tensor encode(const std::vector<std::string>& labels, const std::optional<torch::Device>& device = std::nullopt) const override {
        TODO
    }
    */

    // Define the method myfilter
    std::tuple<torch::Tensor, std::vector<int>> myfilter(const torch::Tensor& probs, const torch::Tensor& ids) const {
        // Convert ids tensor to a vector of ints
        std::vector<int> id_list(ids.data_ptr<int>(), ids.data_ptr<int>() + ids.numel());

        // Find end-of-sequence (EOS) token
        auto eos_it = std::find(id_list.begin(), id_list.end(), eos_id);
        size_t eos_idx = (eos_it != id_list.end()) ? std::distance(id_list.begin(), eos_it) : id_list.size();

        // Slice the id list and probs tensor
        id_list = std::vector<int>(id_list.begin(), id_list.begin() + eos_idx);
        auto probs_slice = probs.slice(0, 0, eos_idx + 1);

        // Return the tuple
        return std::make_tuple(probs_slice, id_list);
    }

private:
    int eos_id;
    int bos_id;
    int pad_id;
};
