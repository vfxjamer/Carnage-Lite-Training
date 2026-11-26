#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace Carnage::Metrics {
	struct TeamEpisodeRewardStats {
		double weightedTotal = 0;
		double weightedAbsoluteTotal = 0;
		std::uint64_t sampleCount = 0;
		std::uint64_t playerCount = 0;

		double MeanPerPlayerStep() const {
			return sampleCount ? weightedTotal / sampleCount : 0;
		}

		double EpisodeTotalPerPlayer() const {
			return playerCount ? weightedTotal / playerCount : 0;
		}
	};

	struct EpisodeRewardStats {
		std::array<TeamEpisodeRewardStats, 2> teams = {};
	};

	inline void RecordPlayerReward(EpisodeRewardStats& episode, int teamIndex, double weightedValue) {
		if (teamIndex < 0 || teamIndex >= static_cast<int>(episode.teams.size()))
			throw std::out_of_range("Team index must be 0 or 1");

		auto& stats = episode.teams[teamIndex];
		stats.weightedTotal += weightedValue;
		stats.weightedAbsoluteTotal += std::abs(weightedValue);
		stats.sampleCount++;
	}

	inline void SetEpisodeTeamPlayerCounts(
		EpisodeRewardStats& episode,
		const std::array<std::uint64_t, 2>& playerCounts
	) {
		for (std::size_t teamIndex = 0; teamIndex < episode.teams.size(); teamIndex++)
			episode.teams[teamIndex].playerCount = playerCounts[teamIndex];
	}
}
