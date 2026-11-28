#pragma once

#include <cstdint>
#include <string_view>

namespace Carnage::Training {
	enum class GameMode {
		OneVOne,
		TwoVTwo
	};

	inline constexpr std::uint64_t kOneVOneTimesteps = 100'000'000'000ULL;
	inline constexpr std::uint64_t kTwoVTwoTimesteps = 20'000'000'000ULL;
	inline constexpr std::uint64_t kCurrentRunTimesteps = 1'000'000'000ULL;
	inline constexpr std::uint64_t kCheckpointTimesteps = 5'000'000ULL;

	constexpr bool IsEnabledPhase(int phase) {
		return phase == 1 || phase == 2;
	}

	constexpr std::uint64_t BoundedRunTimesteps(std::int64_t requestedTimesteps) {
		if (requestedTimesteps <= 0
			|| static_cast<std::uint64_t>(requestedTimesteps) > kCurrentRunTimesteps)
			return kCurrentRunTimesteps;
		return static_cast<std::uint64_t>(requestedTimesteps);
	}

	struct EarlyPhaseRewardWeights {
		float touchBall;
		float speedTowardBall;
		float faceBall;
		float inAir;
		float ballVelocityTowardGoal;
		float goalWeight;
		float concedeScale;
	};

	constexpr EarlyPhaseRewardWeights GetEarlyPhaseRewardWeights(int phase) {
		if (phase == 1)
			return { 50.f, 5.f, 1.f, 0.15f, 0.f, 0.f, -0.8f };
		if (phase == 2)
			return { 1.f, 0.5f, 0.09f, 0.02f, 4.f, 40.f, -0.8f };
		return {};
	}

	constexpr bool IsNumericCheckpointName(std::string_view name) {
		if (name.empty())
			return false;
		for (char character : name)
			if (character < '0' || character > '9')
				return false;
		return true;
	}

	constexpr GameMode ModeAt(std::uint64_t totalTimesteps) {
		return totalTimesteps < kOneVOneTimesteps ? GameMode::OneVOne : GameMode::TwoVTwo;
	}

	constexpr bool CrossedCheckpoint(
		std::uint64_t previousTimesteps,
		std::uint64_t totalTimesteps,
		std::uint64_t checkpointInterval
	) {
		return checkpointInterval > 0 && totalTimesteps > previousTimesteps
			&& totalTimesteps / checkpointInterval > previousTimesteps / checkpointInterval;
	}
}
