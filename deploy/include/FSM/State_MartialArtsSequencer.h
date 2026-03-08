// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// Copyright (c) 2026, Martial Arts Sequencer Extension
// All rights reserved.
//
// Policy Sequencer for Martial Arts Performance
// =============================================
// Chains multiple independently-trained ONNX policies into a continuous
// martial arts performance, similar to the 2026 Spring Festival Gala.
//
// Each "segment" is a (policy, motion_file) pair with a known duration.
// The sequencer plays them back-to-back:
//
//   Segment 0 (heian_shodan) → transition → Segment 1 (front_kick) → ...
//
// Between segments, the robot briefly holds its current pose (blend pause)
// to allow a smooth handoff to the next policy.
//
// Config format (config.yaml):
//   MartialArtsSequencer:
//     transition_hold_s: 1.0        # seconds to hold between segments
//     segments:
//       - { policy_dir: "...", motion_file: "...", fps: 50 }
//       - { policy_dir: "...", motion_file: "...", fps: 50 }

#pragma once

#include "FSM/FSMState.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/observations/motion_observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "LinearInterpolator.h"

#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

struct SequencerSegment
{
    std::filesystem::path policy_dir;
    std::string motion_file;
    float fps;
    float duration_s;  // filled at load time from motion file
};

class State_MartialArtsSequencer : public FSMState
{
public:
    State_MartialArtsSequencer(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        transition_hold_s_ = cfg["transition_hold_s"]
            ? cfg["transition_hold_s"].as<float>() : 1.0f;

        auto segments_cfg = cfg["segments"];
        if (!segments_cfg || !segments_cfg.IsSequence()) {
            throw std::runtime_error("MartialArtsSequencer: 'segments' must be a YAML sequence");
        }

        for (auto it = segments_cfg.begin(); it != segments_cfg.end(); ++it) {
            SequencerSegment seg;

            seg.policy_dir = param::parser_policy_dir((*it)["policy_dir"].as<std::string>());
            seg.motion_file = (*it)["motion_file"].as<std::string>();
            seg.fps = (*it)["fps"] ? (*it)["fps"].as<float>() : 50.0f;

            // Resolve relative motion file path
            if (!std::filesystem::path(seg.motion_file).is_absolute()) {
                seg.motion_file = (param::proj_dir / seg.motion_file).string();
            }

            // Load motion to get duration
            auto motion_data = isaaclab::load_csv(seg.motion_file);
            seg.duration_s = static_cast<float>(motion_data.size()) / seg.fps;

            segments_.push_back(std::move(seg));
        }

        spdlog::info("MartialArtsSequencer: loaded {} segments, total {:.1f}s + {:.1f}s transitions",
            segments_.size(),
            total_motion_duration(),
            transition_hold_s_ * (segments_.size() > 0 ? segments_.size() - 1 : 0));

        // Register safety checks
        this->registered_checks.emplace_back(
            std::make_pair(
                [&]() -> bool { return finished_; },
                FSMStringMap.right.at("FixStand")
            )
        );
        this->registered_checks.emplace_back(
            std::make_pair(
                [&]() -> bool {
                    return current_env_ && isaaclab::mdp::bad_orientation(current_env_.get(), 1.0);
                },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void enter() override
    {
        current_segment_ = 0;
        finished_ = false;
        in_transition_ = false;
        load_segment(0);
        start_policy_thread();
    }

    void run() override
    {
        if (!current_env_) return;

        if (in_transition_ && transition_q_.size() == current_env_->robot->data.joint_ids_map.size()) {
            // Apply interpolated smooth safe poses during transition
            for (int i = 0; i < current_env_->robot->data.joint_ids_map.size(); ++i) {
                 lowcmd->msg_.motor_cmd()[current_env_->robot->data.joint_ids_map[i]].q() = transition_q_[i];
            }
        } 
        else {
            auto action = current_env_->action_manager->processed_actions();
            for (int i = 0; i < current_env_->robot->data.joint_ids_map.size(); ++i) {
                lowcmd->msg_.motor_cmd()[current_env_->robot->data.joint_ids_map[i]].q() = action[i];
            }
        }
    }

    void exit() override
    {
        stop_policy_thread();
        current_env_.reset();
    }

private:
    // ── Segment management ──

    void load_segment(int index)
    {
        if (index >= static_cast<int>(segments_.size())) {
            finished_ = true;
            return;
        }

        const auto& seg = segments_[index];
        spdlog::info("MartialArtsSequencer: loading segment {} — {}",
            index, seg.policy_dir.filename().string());

        auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
            FSMState::lowstate);

        articulation->data.motion_loader = new isaaclab::MotionLoader(
            seg.motion_file, seg.fps);

        current_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
            YAML::LoadFile(seg.policy_dir / "params" / "deploy.yaml"),
            articulation
        );
        current_env_->alg = std::make_unique<isaaclab::OrtRunner>(
            seg.policy_dir / "exported" / "policy.onnx");

        // Set gains
        for (int i = 0; i < current_env_->robot->data.joint_stiffness.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].kp() = current_env_->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = current_env_->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        segment_duration_s_ = seg.duration_s;
    }

    // ── Policy thread ──

    void start_policy_thread()
    {
        policy_thread_running_ = true;
        policy_thread_ = std::thread([this] { policy_loop(); });
    }

    void stop_policy_thread()
    {
        policy_thread_running_ = false;
        if (policy_thread_.joinable()) {
            policy_thread_.join();
        }
    }

    void policy_loop()
    {
        using clock = std::chrono::high_resolution_clock;

        while (policy_thread_running_ && !finished_)
        {
            // ── Execute current segment ──
            current_env_->reset();

            // Compute init yaw alignment (same as State_Mimic)
            auto ref_yaw = isaaclab::yawQuaternion(
                current_env_->robot->data.motion_loader->root_quaternion()).toRotationMatrix();
            auto robot_yaw = isaaclab::yawQuaternion(
                current_env_->robot->data.root_quat_w).toRotationMatrix();
            // Store for observation: init_quat_ = robot_yaw * ref_yaw^T
            // (used by motion_anchor_ori_b observation if needed)

            current_env_->reset();

            const std::chrono::duration<double> step_dur(current_env_->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(step_dur);
            auto sleep_till = clock::now() + dt;

            float elapsed_s = 0.0f;
            while (policy_thread_running_ && elapsed_s < segment_duration_s_)
            {
                current_env_->step();
                elapsed_s += static_cast<float>(current_env_->step_dt);

                std::this_thread::sleep_until(sleep_till);
                sleep_till += dt;
            }

            if (!policy_thread_running_) break;

            // ── Advance to next segment ──
            current_segment_++;
            if (current_segment_ >= static_cast<int>(segments_.size())) {
                spdlog::info("MartialArtsSequencer: performance complete!");
                finished_ = true;
                break;
            }

            // ── Transition hold: smooth interpolation instead of hard jump ──
            if (transition_hold_s_ > 0.0f) {
                spdlog::info("MartialArtsSequencer: smooth transition {:.1f}s to segment {}",
                    transition_hold_s_, current_segment_);
                
                // 1. Capture end state of PREVIOUS segment
                std::vector<float> start_q(current_env_->robot->data.joint_ids_map.size());
                auto prev_action = current_env_->action_manager->processed_actions();
                for(int i=0; i<start_q.size(); ++i) start_q[i] = prev_action[i];

                in_transition_ = true;
                
                // 2. Load NEXT segment safely (it will reset its own env and populate a default start pose)
                load_segment(current_segment_);
                if (!current_env_) break;
                current_env_->reset();
                current_env_->step(); // initial step to get target action for new policy
                
                std::vector<float> target_q(current_env_->robot->data.joint_ids_map.size());
                auto next_action = current_env_->action_manager->processed_actions();
                for(int i=0; i<target_q.size(); ++i) target_q[i] = next_action[i];

                // 3. Linearly interpolate between start_q and target_q
                int hold_steps = static_cast<int>(transition_hold_s_ / current_env_->step_dt);
                for (int step = 0; step <= hold_steps && policy_thread_running_; ++step) {
                    float alpha = static_cast<float>(step) / hold_steps;
                    
                    std::vector<float> interp_q(start_q.size());
                    for(size_t i=0; i<start_q.size(); ++i) {
                        interp_q[i] = start_q[i] + alpha * (target_q[i] - start_q[i]);
                    }
                    
                    // pass to run() loop
                    transition_q_ = interp_q;
                    std::this_thread::sleep_for(std::chrono::milliseconds(int(current_env_->step_dt * 1000)));
                }
                
                in_transition_ = false;
            } else {
                // If no transition hold, just load directly (not recommended)
                if (!policy_thread_running_) break;
                load_segment(current_segment_);
            }
        }
    }

    float total_motion_duration() const
    {
        float total = 0.0f;
        for (const auto& seg : segments_) total += seg.duration_s;
        return total;
    }

    // ── Members ──
    std::vector<SequencerSegment> segments_;
    float transition_hold_s_ = 1.0f;

    int current_segment_ = 0;
    float segment_duration_s_ = 0.0f;
    bool finished_ = false;
    bool in_transition_ = false;
    std::vector<float> transition_q_;

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> current_env_;
    std::thread policy_thread_;
    bool policy_thread_running_ = false;
};

REGISTER_FSM(State_MartialArtsSequencer)
