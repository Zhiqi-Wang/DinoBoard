struct AzulSearchOptions {
    bool stop_on_round_transition = true;
    bool enable_round_transition_chance = true;
    int chance_expand_cap = 10;
    board_ai::search::CommonSearchOptions common{};
};

class SplitMix64Engine {
 public:
    using result_type = std::uint64_t;

    explicit SplitMix64Engine(std::uint64_t seed) : state_(seed) {}

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }

    result_type operator()() {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }

 private:
    std::uint64_t state_;
};

constexpr int kFloorTargetLine = 5;
constexpr std::array<int, 7> kFloorPenalties = {-1, -1, -2, -2, -2, -3, -3};

int wall_col_for_color_heuristic(int row, int color_idx) {
    return (color_idx + row) % kColors;
}

int picked_count_for_action(const AzulState& state, ActionId action) {
    const int source = decode_source(action);
    const int color = decode_color(action);
    if (source == kCenterSource) {
        return static_cast<int>(state.center[color]);
    }
    if (source >= 0 && source < kFactories) {
        return static_cast<int>(state.factories[source][color]);
    }
    return 0;
}

float floor_loss_from_count(int floor_count, int add_count) {
    if (add_count <= 0) {
        return 0.0f;
    }
    const int start = std::max(0, std::min(floor_count, static_cast<int>(kFloorPenalties.size())));
    const int end = std::max(0, std::min(floor_count + add_count, static_cast<int>(kFloorPenalties.size())));
    if (end <= start) {
        return 0.0f;
    }
    int loss = 0;
    for (int i = start; i < end; ++i) {
        loss += -kFloorPenalties[static_cast<size_t>(i)];
    }
    return static_cast<float>(loss);
}

float floor_loss_for_action(const AzulState& state, ActionId action, int explicit_overflow = -1) {
    const int pid = state.current_player;
    const auto& p = state.players[pid];
    const int picked = picked_count_for_action(state, action);
    const int target_line = decode_target_line(action);
    int to_floor = 0;
    if (target_line == kFloorTargetLine) {
        to_floor = picked;
    } else if (explicit_overflow >= 0) {
        to_floor = std::max(0, explicit_overflow);
    } else {
        const int cap = target_line + 1;
        const int free_slots = std::max(0, cap - static_cast<int>(p.line_len[target_line]));
        to_floor = std::max(0, picked - free_slots);
    }
    if (decode_source(action) == kCenterSource && state.first_player_token_in_center) {
        to_floor += 1;
    }
    return floor_loss_from_count(static_cast<int>(p.floor_count), to_floor);
}

float column_centrality(int col) {
    if (col == 2) {
        return 0.55f;
    }
    if (col == 1 || col == 3) {
        return 0.50f;
    }
    return 0.45f;
}

int score_wall_placement_for_mask(std::array<std::uint8_t, kRows> wall_mask, int row, int col) {
    wall_mask[row] = static_cast<std::uint8_t>(wall_mask[row] | (1U << col));
    const auto filled = [&](int r, int c) -> bool {
        if (r < 0 || r >= kRows || c < 0 || c >= kColors) {
            return false;
        }
        return ((wall_mask[r] >> c) & 1U) != 0U;
    };

    int horizontal = 1;
    for (int c = col - 1; c >= 0 && filled(row, c); --c) {
        horizontal += 1;
    }
    for (int c = col + 1; c < kColors && filled(row, c); ++c) {
        horizontal += 1;
    }

    int vertical = 1;
    for (int r = row - 1; r >= 0 && filled(r, col); --r) {
        vertical += 1;
    }
    for (int r = row + 1; r < kRows && filled(r, col); ++r) {
        vertical += 1;
    }

    if (horizontal == 1 && vertical == 1) {
        return 1;
    }
    if (horizontal > 1 && vertical > 1) {
        return horizontal + vertical;
    }
    return std::max(horizontal, vertical);
}

float row45_alignment_bonus(const board_ai::azul::PlayerState& p, int row, int color) {
    const int other_row = (row == 3) ? 4 : 3;
    if (p.line_len[other_row] == 0 || p.line_color[other_row] < 0) {
        return 0.0f;
    }
    const int my_col = wall_col_for_color_heuristic(row, color);
    const int other_col = wall_col_for_color_heuristic(other_row, p.line_color[other_row]);
    return (my_col == other_col) ? 2.2f : -0.9f;
}

float row45_center_column_bonus(int row, int color) {
    const int col = wall_col_for_color_heuristic(row, color);
    return (col >= 1 && col <= 3) ? 1.1f : -0.7f;
}

float late_round_front_row_pressure(const AzulState& state, ActionId action) {
    int remaining_tiles = 0;
    for (int f = 0; f < kFactories; ++f) {
        for (int c = 0; c < kColors; ++c) {
            remaining_tiles += static_cast<int>(state.factories[f][c]);
        }
    }
    for (int c = 0; c < kColors; ++c) {
        remaining_tiles += static_cast<int>(state.center[c]);
    }
    if (remaining_tiles > 8) {
        return 0.0f;
    }
    const int target_line = decode_target_line(action);
    if (target_line < 0 || target_line >= 2) {
        return 0.0f;
    }
    const auto& p = state.players[state.current_player];
    const int need = (target_line + 1) - static_cast<int>(p.line_len[target_line]);
    if (need == 1 && picked_count_for_action(state, action) > 0) {
        return 1.4f;
    }
    return 0.0f;
}

float quick_move_gain(const AzulState& state, ActionId action) {
    const auto& p = state.players[state.current_player];
    const int picked = picked_count_for_action(state, action);
    const int target_line = decode_target_line(action);
    float score = 0.0f;
    bool token_floor_accounted = false;

    if (target_line == kFloorTargetLine) {
        score -= floor_loss_for_action(state, action);
        token_floor_accounted = (decode_source(action) == kCenterSource && state.first_player_token_in_center);
    } else {
        const int cap = target_line + 1;
        const int line_len = static_cast<int>(p.line_len[target_line]);
        const int to_line = std::min(std::max(0, cap - line_len), picked);
        const int overflow = std::max(0, picked - to_line);
        score += static_cast<float>(to_line) * (1.0f + static_cast<float>(target_line) * 0.15f);
        if (line_len + to_line == cap) {
            score += 2.2f + static_cast<float>(target_line) * 0.3f;
        }
        if (overflow > 0) {
            score -= floor_loss_for_action(state, action, overflow);
            if (decode_source(action) == kCenterSource && state.first_player_token_in_center) {
                token_floor_accounted = true;
            }
        }
    }

    if (decode_source(action) == kCenterSource && state.first_player_token_in_center) {
        if (!token_floor_accounted) {
            score -= floor_loss_from_count(static_cast<int>(p.floor_count), 1);
        }
        score += 1.1f;
        if (state.round_index >= 2 && state.round_index <= 4) {
            score += 1.4f;
        }
    }
    return score;
}

float score_heuristic_move(const AzulState& state, ActionId action) {
    const auto& p = state.players[state.current_player];
    const int picked = picked_count_for_action(state, action);
    const int source = decode_source(action);
    const int color = decode_color(action);
    const int target_line = decode_target_line(action);
    float score = 0.0f;
    bool token_floor_accounted = false;

    if (target_line == kFloorTargetLine) {
        score -= floor_loss_for_action(state, action);
        token_floor_accounted = (source == kCenterSource && state.first_player_token_in_center);
    } else {
        const int cap = target_line + 1;
        const int line_len = static_cast<int>(p.line_len[target_line]);
        const int to_line = std::min(std::max(0, cap - line_len), picked);
        const int overflow = std::max(0, picked - to_line);
        const int target_col = wall_col_for_color_heuristic(target_line, color);

        score += static_cast<float>(to_line) * (1.2f + static_cast<float>(target_line) * 0.2f);
        if (line_len + to_line == cap) {
            score += 2.5f + static_cast<float>(target_line) * 0.4f;
        }
        if (overflow > 0) {
            score -= floor_loss_for_action(state, action, overflow);
            if (source == kCenterSource && state.first_player_token_in_center) {
                token_floor_accounted = true;
            }
        }
        score += 0.9f * column_centrality(target_col);

        if (to_line > 0) {
            const int placement_score = score_wall_placement_for_mask(p.wall_mask, target_line, target_col);
            if (line_len + to_line == cap) {
                score += static_cast<float>(placement_score);
            } else {
                const float progress = static_cast<float>(line_len + to_line) / static_cast<float>(cap);
                score += 0.7f * progress * static_cast<float>(placement_score);
            }
        }
        if ((target_line == 0 || target_line == 1) && line_len + to_line == cap) {
            score += 1.8f;
        }
        if ((target_line == 3 || target_line == 4) && line_len == 0 && to_line > 0) {
            score += row45_alignment_bonus(p, target_line, color);
        }
        if ((target_line == 3 || target_line == 4) && to_line > 0) {
            score += row45_center_column_bonus(target_line, color);
        }
    }

    if (source == kCenterSource && state.first_player_token_in_center) {
        int remaining = 0;
        for (int f = 0; f < kFactories; ++f) {
            for (int c = 0; c < kColors; ++c) {
                remaining += static_cast<int>(state.factories[f][c]);
            }
        }
        for (int c = 0; c < kColors; ++c) {
            remaining += static_cast<int>(state.center[c]);
        }
        if (!token_floor_accounted) {
            score -= floor_loss_from_count(static_cast<int>(p.floor_count), 1);
        }
        score += 1.2f;
        if (state.round_index >= 2 && state.round_index <= 4) {
            score += 3.0f;
        }
        if (remaining <= 8) {
            score += 0.6f;
        }
    }

    score += late_round_front_row_pressure(state, action);
    return score;
}

float opponent_pressure_bonus(const AzulState& state, AzulRules& rules, ActionId action) {
    AzulState next = state;
    rules.do_action_fast(next, action);
    if (next.terminal) {
        return 0.0f;
    }
    const auto opp_legal = rules.legal_actions(next);
    if (opp_legal.empty()) {
        return 0.0f;
    }
    float opp_best = -1e9f;
    for (ActionId opp_action : opp_legal) {
        opp_best = std::max(opp_best, quick_move_gain(next, opp_action));
    }
    return -0.45f * opp_best;
}

class AzulNetMctsTraversalLimiter final : public board_ai::search::INetMctsTraversalLimiter {
 public:
    explicit AzulNetMctsTraversalLimiter(const AzulSearchOptions& options) : options_(options) {}

    bool should_stop(
        const board_ai::IGameState& root_state,
        const board_ai::IGameState& current_state,
        int /*depth*/
    ) const override {
        const auto* root = dynamic_cast<const AzulState*>(&root_state);
        const auto* cur = dynamic_cast<const AzulState*>(&current_state);
        if (root == nullptr || cur == nullptr) {
            return false;
        }
        const int round_delta = cur->round_index - root->round_index;
        if (options_.stop_on_round_transition && round_delta > 0) {
            return true;
        }
        return false;
    }

    bool on_truncation_leaf(
        const board_ai::IGameState& root_state,
        const board_ai::IGameState& current_state,
        const board_ai::IGameState* parent_state,
        ActionId parent_action,
        int /*depth*/,
        const board_ai::IGameRules& rules,
        const board_ai::search::IStateValueModel& value_model,
        const board_ai::search::IPolicyValueEvaluator& evaluator,
        float* out_leaf_value) const override {
        if (out_leaf_value == nullptr) {
            return false;
        }
        if (!options_.stop_on_round_transition || !options_.enable_round_transition_chance) {
            return false;
        }
        const auto* root = dynamic_cast<const AzulState*>(&root_state);
        const auto* cur = dynamic_cast<const AzulState*>(&current_state);
        const auto* pre = dynamic_cast<const AzulState*>(parent_state);
        const auto* azul_rules = dynamic_cast<const AzulRules*>(&rules);
        if (root == nullptr || cur == nullptr || pre == nullptr || azul_rules == nullptr) {
            return false;
        }
        if (cur->round_index <= root->round_index) {
            return false;
        }
        if (cur->round_index <= pre->round_index) {
            return false;
        }
        const int cap = std::max(1, options_.chance_expand_cap);
        const std::uint64_t key = chance_key(*pre, parent_action);
        auto& pool = chance_pool_[key];
        float value = 0.0f;
        if (static_cast<int>(pool.size()) < cap) {
            value = sample_transition_value(*pre, parent_action, *azul_rules, value_model, evaluator);
            pool.push_back(value);
        } else {
            const std::size_t idx = static_cast<std::size_t>(chance_rng_() % static_cast<std::uint64_t>(pool.size()));
            value = pool[idx];
        }
        *out_leaf_value = value;
        return true;
    }

 private:
    static std::uint64_t chance_key(const AzulState& pre, ActionId action) {
        std::uint64_t h = static_cast<std::uint64_t>(pre.state_hash(false));
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(action)) + 0x9e3779b97f4a7c15ULL +
             (h << 6U) + (h >> 2U);
        return h;
    }

    static void shuffle_hidden_tiles(AzulState* st, SplitMix64Engine* rng) {
        if (st == nullptr || rng == nullptr) {
            return;
        }
        if (!st->bag.empty()) {
            std::shuffle(st->bag.begin(), st->bag.end(), *rng);
        }
        if (!st->box_lid.empty()) {
            std::shuffle(st->box_lid.begin(), st->box_lid.end(), *rng);
        }
        st->rng_salt ^= static_cast<std::uint64_t>((*rng)());
    }

    static float eval_leaf_value(
        const AzulState& st,
        const AzulRules& rules,
        const board_ai::search::IStateValueModel& value_model,
        const board_ai::search::IPolicyValueEvaluator& evaluator) {
        const int perspective = value_model.current_player(st);
        if (value_model.is_terminal(st)) {
            return value_model.terminal_value_for_player(st, perspective);
        }
        const auto legal = rules.legal_actions(st);
        if (legal.empty()) {
            return value_model.terminal_value_for_player(st, perspective);
        }
        std::vector<float> priors{};
        float value = 0.0f;
        const bool ok = evaluator.evaluate(st, perspective, legal, &priors, &value);
        if (!ok) {
            return 0.0f;
        }
        return value;
    }

    float sample_transition_value(
        const AzulState& pre,
        ActionId action,
        const AzulRules& rules,
        const board_ai::search::IStateValueModel& value_model,
        const board_ai::search::IPolicyValueEvaluator& evaluator) const {
        AzulState sampled = pre;
        shuffle_hidden_tiles(&sampled, &chance_rng_);
        if (!rules.validate_action(sampled, action)) {
            return eval_leaf_value(sampled, rules, value_model, evaluator);
        }
        rules.do_action_fast(sampled, action);
        return eval_leaf_value(sampled, rules, value_model, evaluator);
    }

    AzulSearchOptions options_{};
    mutable std::unordered_map<std::uint64_t, std::vector<float>> chance_pool_{};
    mutable SplitMix64Engine chance_rng_{0xA2B3C4D5E6F77889ULL};
};

bool validate_engine_name(const std::string& engine) {
    if (engine == "heuristic" || engine == "netmcts") {
        return true;
    }
    if (engine == "mcts") {
        PyErr_SetString(PyExc_ValueError, "engine 'mcts' has been removed for azul; use 'heuristic' or 'netmcts'");
        return false;
    }
    PyErr_SetString(PyExc_ValueError, "unsupported engine; expected 'heuristic' or 'netmcts'");
    return false;
}

bool parse_azul_search_options(PyObject* obj, AzulSearchOptions* out) {
    if (!out) return false;
    if (!board_ai::search::parse_common_search_options(obj, &out->common)) return false;
    if (obj == nullptr || obj == Py_None) return true;

    if (!board_ai::search::parse_bool_option(
            obj,
            "stop_on_round_transition",
            &out->stop_on_round_transition)) {
        return false;
    }
    if (!board_ai::search::parse_int_option(
            obj,
            "chance_expand_cap",
            &out->chance_expand_cap,
            1)) {
        return false;
    }
    if (!board_ai::search::parse_bool_option(
            obj,
            "enable_round_transition_chance",
            &out->enable_round_transition_chance)) {
        return false;
    }
    if (!board_ai::search::parse_int_option(
            obj,
            "max_search_depth",
            &out->common.max_search_depth,
            1)) {
        return false;
    }
    if (!board_ai::search::parse_int_option(
            obj,
            "max_episode_plies",
            &out->common.max_episode_plies,
            1)) {
        return false;
    }
    return true;
}

ActionId choose_action_for_state(
    AzulState& state,
    AzulRules& rules,
    const std::string& engine,
    int simulations,
    double temperature,
    int time_budget_ms,
    const char* model_path,
    const AzulSearchOptions& search_options,
    int ply_index = 0,
    std::vector<ActionId>* out_policy_actions = nullptr,
    std::vector<int>* out_policy_visits = nullptr
) {
    struct ThreadLocalEvaluatorCache {
        std::string model_path{};
        std::unique_ptr<AzulFeatureEncoder> encoder{};
        std::unique_ptr<OnnxPolicyValueEvaluator> evaluator{};
    };
    static thread_local ThreadLocalEvaluatorCache eval_cache{};

    if (out_policy_actions != nullptr) {
        out_policy_actions->clear();
    }
    if (out_policy_visits != nullptr) {
        out_policy_visits->clear();
    }
    const auto legal = rules.legal_actions(state);
    if (legal.empty()) {
        return -1;
    }
    ActionId best = legal.front();
    if (engine == "heuristic") {
        std::vector<std::pair<float, ActionId>> scored;
        scored.reserve(legal.size());
        for (ActionId a : legal) {
            scored.emplace_back(score_heuristic_move(state, a), a);
        }
        std::sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
            }
            return lhs.first > rhs.first;
        });

        const size_t top_k = std::min<size_t>(8, scored.size());
        for (size_t i = 0; i < top_k; ++i) {
            scored[i].first += opponent_pressure_bonus(state, rules, scored[i].second);
        }
        std::sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
            }
            return lhs.first > rhs.first;
        });
        if (temperature <= 1e-6 || scored.size() == 1) {
            best = scored.front().second;
        } else {
            const double inv_t = 1.0 / std::max(1e-6, temperature);
            double max_logit = -std::numeric_limits<double>::infinity();
            for (const auto& it : scored) {
                max_logit = std::max(max_logit, static_cast<double>(it.first) * inv_t);
            }
            std::vector<double> probs(scored.size(), 0.0);
            double sum = 0.0;
            for (size_t i = 0; i < scored.size(); ++i) {
                const double w = std::exp(static_cast<double>(scored[i].first) * inv_t - max_logit);
                probs[i] = std::isfinite(w) ? w : 0.0;
                sum += probs[i];
            }
            if (sum <= 1e-12) {
                best = scored.front().second;
            } else {
                for (double& p : probs) {
                    p /= sum;
                }
                SplitMix64Engine rng(
                    static_cast<std::uint64_t>(state.rng_salt) ^
                    static_cast<std::uint64_t>(state.round_index + 1)
                );
                std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
                best = scored[dist(rng)].second;
            }
        }
    } else if (engine == "netmcts") {
        if (model_path == nullptr || std::string(model_path).empty()) {
            PyErr_SetString(PyExc_RuntimeError, "netmcts requires non-empty model_path");
            return -1;
        }
        const std::string requested_model_path(model_path);
        NetMctsConfig cfg;
        cfg.simulations = std::max(1, simulations);
        cfg.max_depth = std::max(1, search_options.common.max_search_depth);
        cfg.c_puct = 1.4f;
        cfg.value_clip = 1.0f;
        const auto noise = board_ai::search::resolve_root_dirichlet_noise(
            search_options.common.dirichlet_alpha,
            search_options.common.dirichlet_epsilon,
            search_options.common.dirichlet_on_first_n_plies,
            ply_index
        );
        cfg.root_dirichlet_alpha = noise.alpha;
        cfg.root_dirichlet_epsilon = noise.epsilon;
        AzulNetMctsTraversalLimiter limiter(search_options);
        cfg.traversal_limiter = &limiter;
        if (!eval_cache.evaluator || eval_cache.model_path != requested_model_path) {
            eval_cache.model_path = requested_model_path;
            eval_cache.encoder = std::make_unique<AzulFeatureEncoder>();
            eval_cache.evaluator =
                std::make_unique<OnnxPolicyValueEvaluator>(eval_cache.model_path, eval_cache.encoder.get());
        }
        if (!eval_cache.evaluator || !eval_cache.evaluator->is_ready()) {
            const std::string err =
                (eval_cache.evaluator && !eval_cache.evaluator->last_error().empty())
                    ? eval_cache.evaluator->last_error()
                    : "unknown evaluator init error";
            PyErr_SetString(
                PyExc_RuntimeError,
                ("netmcts evaluator is not ready (model may be ignored): " + err).c_str()
            );
            return -1;
        }
        AzulStateValueModel value_model;
        NetMctsStats net_stats;
        NetMcts net(cfg);
        best = net.search_root(state, rules, value_model, *eval_cache.evaluator, &net_stats);
        if (out_policy_actions != nullptr && out_policy_visits != nullptr &&
            !net_stats.root_actions.empty() &&
            net_stats.root_actions.size() == net_stats.root_action_visits.size()) {
            out_policy_actions->reserve(net_stats.root_actions.size());
            out_policy_visits->reserve(net_stats.root_action_visits.size());
            for (size_t i = 0; i < net_stats.root_action_visits.size(); ++i) {
                out_policy_actions->push_back(net_stats.root_actions[i]);
                out_policy_visits->push_back(std::max(0, net_stats.root_action_visits[i]));
            }
        }
        if (!net_stats.root_actions.empty() &&
            net_stats.root_actions.size() == net_stats.root_action_visits.size()) {
            const std::uint64_t sample_seed = static_cast<std::uint64_t>(state.rng_salt) ^
                                              static_cast<std::uint64_t>(state.round_index + 1) ^
                                              static_cast<std::uint64_t>(state.current_player + 7);
            best = select_action_from_visits(
                net_stats.root_actions,
                net_stats.root_action_visits,
                temperature,
                sample_seed,
                best
            );
        }
        if (best < 0) {
            best = legal.front();
        }
    } else {
        (void)time_budget_ms;
        return -1;
    }
    return best;
}

PyObject* py_session_ai_move(PyObject*, PyObject* args) {
    PyObject* h = nullptr;
    const char* engine = "netmcts";
    int simulations = 200;
    double temperature = 1.0;
    int time_budget_ms = 0;
    const char* model_path = nullptr;
    PyObject* search_options = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "Osidi|zO",
            &h,
            &engine,
            &simulations,
            &temperature,
            &time_budget_ms,
            &model_path,
            &search_options
        )) {
        return nullptr;
    }
    AzulSearchOptions parsed_options{};
    if (!parse_azul_search_options(search_options, &parsed_options)) {
        return nullptr;
    }
    if (!validate_engine_name(std::string(engine))) {
        return nullptr;
    }
    Session* s = get_session_from_handle(h);
    if (!s) return nullptr;
    ActionId best = choose_action_for_state(
        s->state, s->rules, std::string(engine), simulations, temperature, time_budget_ms, model_path, parsed_options, 0
    );
    if (best < 0) {
        if (PyErr_Occurred()) {
            return nullptr;
        }
        Py_RETURN_NONE;
    }
    return apply_action_common(*s, best, false);
}

PyObject* py_run_selfplay_episode_fast(PyObject*, PyObject* args) {
    long long seed = 0;
    const char* engine = "netmcts";
    int simulations = 200;
    double temperature = 1.0;
    const char* model_path = nullptr;
    PyObject* search_options = nullptr;
    if (!PyArg_ParseTuple(args, "Lsid|zO", &seed, &engine, &simulations, &temperature, &model_path, &search_options)) {
        return nullptr;
    }
    AzulSearchOptions parsed_options{};
    if (!parse_azul_search_options(search_options, &parsed_options)) {
        return nullptr;
    }
    if (!validate_engine_name(std::string(engine))) {
        return nullptr;
    }

    AzulState state;
    AzulRules rules;
    AzulFeatureEncoder encoder;
    state.reset_with_seed(static_cast<std::uint64_t>(seed));
    int cursor = 0;

    PyObject* samples = PyList_New(0);
    if (!samples) return nullptr;

    while (!state.terminal && cursor < std::max(1, parsed_options.common.max_episode_plies)) {
        const auto legal = rules.legal_actions(state);
        if (legal.empty()) {
            break;
        }
        const int current_player = state.current_player;
        std::vector<float> features;
        std::vector<float> legal_mask;
        if (!encoder.encode(state, current_player, legal, &features, &legal_mask)) {
            features.clear();
        }
        std::vector<ActionId> policy_actions;
        std::vector<int> policy_visits;
        ActionId action =
            choose_action_for_state(
                state, rules, std::string(engine), simulations, temperature, 0, model_path, parsed_options,
                cursor, &policy_actions, &policy_visits
            );
        if (action < 0) {
            if (PyErr_Occurred()) {
                Py_DECREF(samples);
                return nullptr;
            }
            break;
        }
        rules.do_action_fast(state, action);
        cursor += 1;

        PyObject* rec = PyDict_New();
        if (!rec) {
            Py_DECREF(samples);
            return nullptr;
        }
        PyDict_SetItemString(rec, "ply", PyLong_FromLong(cursor));
        PyDict_SetItemString(rec, "player", PyLong_FromLong(current_player));
        PyDict_SetItemString(rec, "action_id", PyLong_FromLong(action));
        PyDict_SetItemString(rec, "state_version", PyLong_FromLong(cursor - 1));
        if (!policy_actions.empty() && policy_actions.size() == policy_visits.size()) {
            PyObject* pa_list = PyList_New(static_cast<Py_ssize_t>(policy_actions.size()));
            PyObject* pv_list = PyList_New(static_cast<Py_ssize_t>(policy_visits.size()));
            if (!pa_list || !pv_list) {
                Py_XDECREF(pa_list);
                Py_XDECREF(pv_list);
                Py_DECREF(rec);
                Py_DECREF(samples);
                return nullptr;
            }
            for (size_t i = 0; i < policy_actions.size(); ++i) {
                PyObject* aid = PyLong_FromLong(policy_actions[i]);
                PyObject* visits = PyLong_FromLong(policy_visits[i]);
                if (!aid || !visits) {
                    Py_XDECREF(aid);
                    Py_XDECREF(visits);
                    Py_DECREF(pa_list);
                    Py_DECREF(pv_list);
                    Py_DECREF(rec);
                    Py_DECREF(samples);
                    return nullptr;
                }
                PyList_SET_ITEM(pa_list, static_cast<Py_ssize_t>(i), aid);
                PyList_SET_ITEM(pv_list, static_cast<Py_ssize_t>(i), visits);
            }
            PyDict_SetItemString(rec, "policy_action_ids", pa_list);
            PyDict_SetItemString(rec, "policy_action_visits", pv_list);
            Py_DECREF(pa_list);
            Py_DECREF(pv_list);
        }
        PyObject* feat_list = PyList_New(static_cast<Py_ssize_t>(features.size()));
        if (!feat_list) {
            Py_DECREF(rec);
            Py_DECREF(samples);
            return nullptr;
        }
        for (size_t i = 0; i < features.size(); ++i) {
            PyObject* fv = PyFloat_FromDouble(static_cast<double>(features[i]));
            if (!fv) {
                Py_DECREF(feat_list);
                Py_DECREF(rec);
                Py_DECREF(samples);
                return nullptr;
            }
            PyList_SET_ITEM(feat_list, static_cast<Py_ssize_t>(i), fv);
        }
        PyDict_SetItemString(rec, "features", feat_list);
        Py_DECREF(feat_list);
        PyList_Append(samples, rec);
        Py_DECREF(rec);
    }
    if (!state.terminal && cursor >= std::max(1, parsed_options.common.max_episode_plies)) {
        state.terminal = true;
        state.winner = -1;
        state.shared_victory = true;
    }

    PyObject* out = PyDict_New();
    if (!out) {
        Py_DECREF(samples);
        return nullptr;
    }
    PyDict_SetItemString(out, "samples", samples);
    Py_DECREF(samples);
    PyDict_SetItemString(out, "plies", PyLong_FromLong(cursor));
    if (!state.terminal || state.winner < 0) {
        Py_INCREF(Py_None);
        PyDict_SetItemString(out, "winner", Py_None);
    } else {
        PyDict_SetItemString(out, "winner", PyLong_FromLong(state.winner));
    }
    PyDict_SetItemString(out, "shared_victory", state.shared_victory ? Py_True : Py_False);
    PyObject* scores = PyList_New(2);
    PyList_SET_ITEM(scores, 0, PyLong_FromLong(state.scores[0]));
    PyList_SET_ITEM(scores, 1, PyLong_FromLong(state.scores[1]));
    PyDict_SetItemString(out, "scores", scores);
    Py_DECREF(scores);
    return out;
}

PyObject* py_run_arena_match_fast(PyObject*, PyObject* args) {
    long long seed = 0;
    const char* p0_engine = "netmcts";
    int p0_simulations = 200;
    double p0_temperature = 1.0;
    const char* p0_model_path = nullptr;
    const char* p1_engine = "netmcts";
    int p1_simulations = 200;
    double p1_temperature = 1.0;
    const char* p1_model_path = nullptr;
    PyObject* p0_search_options = nullptr;
    PyObject* p1_search_options = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "Lsidzsidz|OO",
            &seed,
            &p0_engine,
            &p0_simulations,
            &p0_temperature,
            &p0_model_path,
            &p1_engine,
            &p1_simulations,
            &p1_temperature,
            &p1_model_path,
            &p0_search_options,
            &p1_search_options
        )) {
        return nullptr;
    }
    AzulSearchOptions parsed_p0_options{};
    AzulSearchOptions parsed_p1_options{};
    if (!parse_azul_search_options(p0_search_options, &parsed_p0_options)) {
        return nullptr;
    }
    if (!parse_azul_search_options(p1_search_options, &parsed_p1_options)) {
        return nullptr;
    }
    if (!validate_engine_name(std::string(p0_engine)) || !validate_engine_name(std::string(p1_engine))) {
        return nullptr;
    }

    AzulState state;
    AzulRules rules;
    state.reset_with_seed(static_cast<std::uint64_t>(seed));
    int cursor = 0;
    const int max_episode_plies =
        std::max(1, std::min(parsed_p0_options.common.max_episode_plies, parsed_p1_options.common.max_episode_plies));
    while (!state.terminal && cursor < max_episode_plies) {
        const bool p0_turn = state.current_player == 0;
        const std::string engine = p0_turn ? std::string(p0_engine) : std::string(p1_engine);
        const int simulations = p0_turn ? p0_simulations : p1_simulations;
        const char* model_path = p0_turn ? p0_model_path : p1_model_path;
        const auto& search_options = p0_turn ? parsed_p0_options : parsed_p1_options;
        const double temperature = p0_turn ? p0_temperature : p1_temperature;
        ActionId action =
            choose_action_for_state(state, rules, engine, simulations, temperature, 0, model_path, search_options, cursor);
        if (action < 0) {
            if (PyErr_Occurred()) {
                return nullptr;
            }
            break;
        }
        rules.do_action_fast(state, action);
        cursor += 1;
    }
    if (!state.terminal && cursor >= max_episode_plies) {
        state.terminal = true;
        state.winner = -1;
        state.shared_victory = true;
    }

    PyObject* out = PyDict_New();
    if (!out) return nullptr;
    if (!state.terminal || state.winner < 0) {
        Py_INCREF(Py_None);
        PyDict_SetItemString(out, "winner", Py_None);
    } else {
        PyDict_SetItemString(out, "winner", PyLong_FromLong(state.winner));
    }
    PyDict_SetItemString(out, "shared_victory", state.shared_victory ? Py_True : Py_False);
    PyObject* scores = PyList_New(2);
    PyList_SET_ITEM(scores, 0, PyLong_FromLong(state.scores[0]));
    PyList_SET_ITEM(scores, 1, PyLong_FromLong(state.scores[1]));
    PyDict_SetItemString(out, "scores", scores);
    Py_DECREF(scores);
    return out;
}
