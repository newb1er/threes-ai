/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and
 * environments
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#include "action.h"
#include "board.h"
#include "utils.h"
#include "weight.h"

class agent {
 public:
  agent(const std::string& args = "") {
    std::stringstream ss("name=unknown role=unknown " + args);
    for (std::string pair; ss >> pair;) {
      std::string key = pair.substr(0, pair.find('='));
      std::string value = pair.substr(pair.find('=') + 1);
      meta[key] = {value};
    }
  }
  virtual ~agent() {}
  virtual void open_episode(const std::string& flag = "") {}
  virtual void close_episode(const std::string& flag = "") {}
  virtual action take_action(const board& b) { return action(); }
  virtual bool check_for_win(const board& b) { return false; }

 public:
  virtual std::string property(const std::string& key) const {
    return meta.at(key);
  }
  virtual void notify(const std::string& msg) {
    meta[msg.substr(0, msg.find('='))] = {msg.substr(msg.find('=') + 1)};
  }
  virtual std::string name() const { return property("name"); }
  virtual std::string role() const { return property("role"); }

 protected:
  typedef std::string key;
  struct value {
    std::string value;
    operator std::string() const { return value; }
    template <typename numeric,
              typename = typename std::enable_if<
                  std::is_arithmetic<numeric>::value, numeric>::type>
    operator numeric() const {
      return numeric(std::stod(value));
    }
  };
  std::map<key, value> meta;
};

/**
 * base agent playing in heuristic style
 */
class merge_larger_agent : public agent {
 private:
  static const unsigned ONETWO_SCORE = 5;
  static const unsigned SPACE_SCORE = 1;

 public:
  merge_larger_agent(const std::string& args = "") : agent(args) {}

  /**
   * merge larger pile first
   * slide priority: left > up > right > down
   */
  virtual action take_action(const board& b) {
    board _b = board(b);

    unsigned horizontal_score = merge_larger(_b);
    unsigned vertical_score = merge_larger(_b, true);

    // std::cerr << "horizontal: " << horizontal_score << '\n';
    // std::cerr << "vertical: " << vertical_score << '\n';

    if (horizontal_score >= vertical_score && _b.slide(board::LEFT) != -1) {
      return action::slide(board::LEFT);
    } else if (horizontal_score < vertical_score && _b.slide(board::UP) != -1) {
      return action::slide(board::UP);
    } else if (_b.slide(board::RIGHT) != -1) {
      return action::slide(board::RIGHT);
    } else if (_b.slide(board::DOWN) != -1) {
      return action::slide(board::DOWN);
    }

    return action();
  }

 private:
  unsigned merge_larger(board& b, bool transpose = false) {
    if (transpose) b.transpose();

    unsigned space = 0;
    unsigned score = 0;

    // horizontal merge
    for (unsigned r = 0; r < 4; ++r) {
      board::row _row = b[r];
      board::cell pivot = _row[0];

      for (unsigned c = 1; c < 4; ++c) {
        if (_row[c] == 0) { /* if empty, skip */
          space = SPACE_SCORE;
          continue;
        }

        if (pivot == 0) { /* if pivot is empty, swap */
          pivot = _row[c];
          continue;
        }

        if (_row[c] + pivot == 3) { /* a pair of 1 and 2*/
          score += ONETWO_SCORE;

          if (c < 3) {
            pivot = _row[c + 1];
            c++;
          }
          continue;
        }

        if (_row[c] > 2 && pivot > 2 &&
            _row[c] == pivot) { /* not a pair of 1 and 2*/
          score += pivot;

          if (c < 3) {
            pivot = _row[c + 1];
            c++;
          }
        } else {
          pivot = _row[c];
        }
      }
    }

    if (transpose) b.transpose();

    return score + space;
  }
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
 public:
  random_agent(const std::string& args = "") : agent(args) {
    if (meta.find("seed") != meta.end()) engine.seed(int(meta["seed"]));
  }
  virtual ~random_agent() {}

 protected:
  std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
 public:
  weight_agent(const std::string& args = "") : agent(args), alpha(0) {
    if (meta.find("init") != meta.end()) init_weights(meta["init"]);
    if (meta.find("load") != meta.end()) load_weights(meta["load"]);
    if (meta.find("alpha") != meta.end()) alpha = float(meta["alpha"]);
  }
  virtual ~weight_agent() {
    if (meta.find("save") != meta.end()) save_weights(meta["save"]);
  }

 protected:
  virtual void init_weights(const std::string& info) {
    std::string res = info;  // comma-separated sizes, e.g., "65536,65536"
    for (char& ch : res)
      if (!std::isdigit(ch)) ch = ' ';
    std::stringstream in(res);
    for (size_t size; in >> size; net.emplace_back(size))
      ;
  }
  virtual void load_weights(const std::string& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) std::exit(-1);
    uint32_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    net.resize(size);
    for (weight& w : net) in >> w;
    in.close();
  }
  virtual void save_weights(const std::string& path) {
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) std::exit(-1);
    uint32_t size = net.size();
    out.write(reinterpret_cast<char*>(&size), sizeof(size));
    for (weight& w : net) out << w;
    out.close();
  }

 protected:
  std::vector<weight> net;
  float alpha;
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
 public:
  random_placer(const std::string& args = "")
      : random_agent("name=place role=placer " + args) {
    spaces[0] = {12, 13, 14, 15};
    spaces[1] = {0, 4, 8, 12};
    spaces[2] = {0, 1, 2, 3};
    spaces[3] = {3, 7, 11, 15};
    spaces[4] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  }

  virtual action take_action(const board& after) {
    std::vector<int> space = spaces[after.last()];
    std::shuffle(space.begin(), space.end(), engine);
    for (int pos : space) {
      if (after(pos) != 0) continue;

      int bag[3], num = 0;
      for (board::cell t = 1; t <= 3; t++)
        for (size_t i = 0; i < after.bag(t); i++) bag[num++] = t;
      std::shuffle(bag, bag + num, engine);

      board::cell tile = after.hint() ?: bag[--num];
      board::cell hint = bag[--num];

      return action::place(pos, tile, hint);
    }
    return action();
  }

 private:
  std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
 public:
  random_slider(const std::string& args = "")
      : random_agent("name=slide role=slider " + args), opcode({0, 1, 2, 3}) {}

  virtual action take_action(const board& before) {
    std::shuffle(opcode.begin(), opcode.end(), engine);
    for (int op : opcode) {
      board::reward reward = board(before).slide(op);
      if (reward != -1) return action::slide(op);
    }
    return action();
  }

 private:
  std::array<int, 4> opcode;
};

class ntuple_slider : public weight_agent {
 public:
  ntuple_slider(const std::string& args) : weight_agent(args) {
    tuple_n = net.size();

    prev_reward = 0;
  }

  void set_encoding(std::vector<std::vector<unsigned>>&& e) { encodings = e; }

  virtual void open_episode(const std::string& flag = "") {
    prev_reward = 0;
    t.clear();
    t.reserve(1000);
  }

  virtual void close_episode(const std::string& flag = "") {
    float r = 0.0f;
    float next_value = 0.0f;

    for (auto iter = t.rbegin(); iter != t.rend(); ++iter) {
      auto [current_state, _r] = *iter;

      float current_value = 0.0f;
      for (auto& w : current_state) current_value += *w;

      auto loss = r + next_value - current_value;

      /*
      std::cout << "=============\n";
      std::cout << "current_value: " << current_value << '\n';
      std::cout << "reward: " << _r << '\n';
      std::cout << "loss: " << loss << '\n';
      std::cout << "=============\n";
      */

      for (auto& w : current_state) *w += alpha * loss;

      next_value = current_value;
      r = _r;
    }
  }

  virtual action take_action(const board& b) {
    auto weights = get_weights(b);
    std::vector<weight::type> rewards;
    rewards.reserve(4);

    auto value = 0;

#pragma omp simd
    for (size_t i = 0; i < weights.size(); ++i) value += *weights[i];

    for (int i = 0; i < 4; ++i) {
      board _b(b);
      auto _r = _b.slide(i);

      if (_r == -1) {
        rewards.push_back(std::numeric_limits<float>::lowest());
        continue;
      }

      auto _w = get_weights(_b);
      float _v = 0;
      for (auto& w : _w) _v += *w;

      rewards.push_back(static_cast<float>(reward_fn(_r)) + _v);
    }

    size_t best_action = argmax(rewards.begin(), rewards.end());
    auto r = board(b).slide(best_action);

    // auto loss = (r == -1 ? 0 : rewards.at(best_action)) - value;
    // for (auto w : weights) *w += alpha * loss;

    //prev_reward = board(b).slide(best_action);

    board _b(b);
    _b.slide(best_action);

    if (r != -1) {
      t.emplace_back(get_weights(_b), reward_fn(r));
    }

    /*
    if (board(b).slide(best_action) == -1) {
      std::cerr << "Invalid action.\n";
      auto idx = 0;

      std::cerr << "Best action: " << best_action << '\n';

      std::cerr << "Current Board\n" << board(b) << '\n';

      for (auto& r : rewards) {
        std::cerr << "reward " << idx << ": " << r;
        std::cerr << " | actual reward :" << board(b).slide(idx) << '\n';
        ++idx;
      }

      std::cin >> idx;
    }
    */

    return r == -1 ? action() : action::slide(best_action);
  }

 private:
  std::vector<std::vector<unsigned>> encodings;
  size_t tuple_n;
  unsigned entity_size;
  board::reward prev_reward;
  std::vector<std::pair<std::vector<weight::type*>, float>> t;

  board::reward reward_fn(board::reward& r) {
    return 1 << static_cast<int>(std::floor(std::log(r + 1))) << 5;
  }

  std::vector<weight::type*> get_weights(const board& b) {
    std::vector<weight::type*> w;
    w.reserve(tuple_n);

    for (auto i = 0; i < encodings.size(); ++i) {
      const auto enc = encodings.at(i);
      uint32_t idx = 0;

      for (auto& e : enc) {
        idx = (idx << 4) | b(e);
      }

      w.push_back(&net[i % 2][idx]);
    }

    return w;
  }
};
