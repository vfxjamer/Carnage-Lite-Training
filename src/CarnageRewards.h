#pragma once

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/Reward.h>

#include "RewardEpisodeMetrics.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace RLGC {
	class SpeedTowardBallReward : public Reward {
	public:
		float GetReward(const Player& player, const GameState& state, bool) override {
			const Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			return RS_MAX(0, dirToBall.Dot(player.vel)) / CommonValues::CAR_MAX_SPEED;
		}
	};

	class TouchBallTickDecayReward : public Reward {
		std::unordered_map<std::uint32_t, std::uint64_t> previousTouchTicks;

	public:
		void Reset(const GameState&) override {
			previousTouchTicks.clear();
		}

		float GetReward(const Player& player, const GameState&, bool) override {
			if (!player.ballTouchedStep)
				return 0.f;

			const std::uint64_t touchTick = player.ballHitInfo.tickCountWhenHit;
			const auto previousTouch = previousTouchTicks.find(player.carId);
			if (previousTouch == previousTouchTicks.end()) {
				previousTouchTicks.emplace(player.carId, touchTick);
				return 1.f;
			}

			const auto ticksSincePreviousTouch = touchTick - previousTouch->second;
			previousTouch->second = touchTick;
			return RS_CLAMP(static_cast<float>(ticksSincePreviousTouch) / 120.f, 0.f, 1.f);
		}
	};

	class AirTouchReward : public Reward {
	public:
		float GetReward(const Player& player, const GameState& state, bool) override {
			if (!player.ballTouchedStep || player.isOnGround)
				return 0;

			const float airTimeFraction = RS_CLAMP(player.airTime / 1.75f, 0.f, 1.f);
			const float heightFraction = RS_CLAMP(state.ball.pos.z / CommonValues::CEILING_Z, 0.f, 1.f);
			return RS_MIN(airTimeFraction, heightFraction);
		}
	};

	class DemoBumpReward : public Reward {
	public:
		float GetReward(const Player& player, const GameState&, bool) override {
			return player.eventState.demo || player.eventState.bump ? 1.f : 0.f;
		}
	};

	class EpisodeMetricReward : public Reward {
		Reward* child;
		float weight;
		std::string name;
		Carnage::Metrics::EpisodeRewardStats episodeStats;

	public:
		EpisodeMetricReward(std::string name, Reward* child, float weight)
			: child(child), weight(weight), name(std::move(name)) {}

		~EpisodeMetricReward() override {
			delete child;
		}

		void Reset(const GameState& initialState) override {
			child->Reset(initialState);
			episodeStats = {};
		}

		void PreStep(const GameState& state) override {
			child->PreStep(state);
		}

		float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return child->GetReward(player, state, isFinal);
		}

		std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
			auto values = child->GetAllRewards(state, isFinal);
			std::array<std::uint64_t, 2> playerCounts = {};
			for (std::size_t playerIndex = 0; playerIndex < values.size(); playerIndex++) {
				const int teamIndex = static_cast<int>(state.players[playerIndex].team);
				const float value = values[playerIndex];
				const double weightedValue = value * weight;
				Carnage::Metrics::RecordPlayerReward(episodeStats, teamIndex, weightedValue);
				playerCounts.at(static_cast<std::size_t>(teamIndex))++;
			}
			Carnage::Metrics::SetEpisodeTeamPlayerCounts(episodeStats, playerCounts);
			return values;
		}

		std::string GetName() override {
			return name;
		}

		Carnage::Metrics::EpisodeRewardStats TakeEpisodeStats() {
			Carnage::Metrics::EpisodeRewardStats result = episodeStats;
			episodeStats = {};
			return result;
		}
	};
}
