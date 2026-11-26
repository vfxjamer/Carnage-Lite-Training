#include "TrainingSchedule.h"

#include <cmath>
#include <iostream>

#define CHECK(expression) do { \
	if (!(expression)) { \
		std::cerr << "Check failed at line " << __LINE__ << ": " #expression << '\n'; \
		return 1; \
	} \
} while (false)

int main() {
	using namespace Carnage::Training;

	CHECK(IsEnabledPhase(1));
	CHECK(IsEnabledPhase(2));
	CHECK(!IsEnabledPhase(0));
	CHECK(!IsEnabledPhase(3));
	CHECK(!IsEnabledPhase(5));
	CHECK(BoundedRunTimesteps(0) == kCurrentRunTimesteps);
	CHECK(BoundedRunTimesteps(500'000'000) == 500'000'000ULL);
	CHECK(BoundedRunTimesteps(2'000'000'000) == kCurrentRunTimesteps);
	const auto phaseOneRewards = GetEarlyPhaseRewardWeights(1);
	CHECK(phaseOneRewards.touchBall == 50.f);
	CHECK(phaseOneRewards.goalWeight == 0.f);
	const auto phaseTwoRewards = GetEarlyPhaseRewardWeights(2);
	CHECK(phaseTwoRewards.touchBall == 5.f);
	CHECK(phaseTwoRewards.goalWeight == 30.f);
	CHECK(phaseTwoRewards.concedeScale == -0.8f);
	CHECK(phaseTwoRewards.goalWeight * phaseTwoRewards.concedeScale == -24.f);

	CHECK(ModeAt(0) == GameMode::OneVOne);
	CHECK(ModeAt(kOneVOneTimesteps - 1) == GameMode::OneVOne);
	CHECK(ModeAt(kOneVOneTimesteps) == GameMode::TwoVTwo);
	CHECK(IsNumericCheckpointName("5000000"));
	CHECK(!IsNumericCheckpointName("5000000.tmp"));
	CHECK(!IsNumericCheckpointName(""));

	CHECK(TeamSpiritAt(kOneVOneTimesteps, kOneVOneTimesteps) == 0.0f);
	CHECK(std::abs(TeamSpiritAt(kOneVOneTimesteps + kTwoVTwoTimesteps / 2,
		kOneVOneTimesteps) - 0.5f) < 0.0001f);
	CHECK(TeamSpiritAt(kOneVOneTimesteps + kTwoVTwoTimesteps,
		kOneVOneTimesteps) == 1.0f);
	CHECK(TeamSpiritAt(kOneVOneTimesteps + kTwoVTwoTimesteps * 2,
		kOneVOneTimesteps) == 1.0f);

	CHECK(!CrossedCheckpoint(0, kCheckpointTimesteps - 1, kCheckpointTimesteps));
	CHECK(CrossedCheckpoint(kCheckpointTimesteps - 1,
		kCheckpointTimesteps, kCheckpointTimesteps));
	CHECK(CrossedCheckpoint(kCheckpointTimesteps - 100,
		kCheckpointTimesteps + 100, kCheckpointTimesteps));
	CHECK(!CrossedCheckpoint(kCheckpointTimesteps + 1,
		kCheckpointTimesteps * 2 - 1, kCheckpointTimesteps));
	return 0;
}
