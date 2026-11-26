#include "RewardEpisodeMetrics.h"

#include <cmath>
#include <iostream>

#define CHECK(expression) do { \
	if (!(expression)) { \
		std::cerr << "Check failed at line " << __LINE__ << ": " #expression << '\n'; \
		return 1; \
	} \
} while (false)

int main() {
	using namespace Carnage::Metrics;

	EpisodeRewardStats episode;
	RecordPlayerReward(episode, 0, 2.5);
	RecordPlayerReward(episode, 0, 1.5);
	RecordPlayerReward(episode, 1, -2.5);
	RecordPlayerReward(episode, 1, -1.5);
	SetEpisodeTeamPlayerCounts(episode, { 2, 2 });

	const auto& blue = episode.teams[0];
	const auto& orange = episode.teams[1];
	CHECK(std::abs(blue.MeanPerPlayerStep() - 2.0) < 1e-9);
	CHECK(std::abs(orange.MeanPerPlayerStep() + 2.0) < 1e-9);
	CHECK(std::abs(blue.EpisodeTotalPerPlayer() - 2.0) < 1e-9);
	CHECK(std::abs(orange.EpisodeTotalPerPlayer() + 2.0) < 1e-9);
	CHECK(std::abs(blue.weightedAbsoluteTotal - orange.weightedAbsoluteTotal) < 1e-9);
	return 0;
}
