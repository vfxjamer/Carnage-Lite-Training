#pragma once

#include <RLGymCPP/Math.h>
#include <RLGymCPP/StateSetters/StateSetter.h>

#include <algorithm>

namespace RLGC {
	class ProceduralMidGameState : public StateSetter {
	public:
		void ResetArena(Arena* arena) override {
			arena->ResetToRandomKickoff();

			const Vec ballPos(
				RocketSim::Math::RandFloat(-2200, 2200),
				RocketSim::Math::RandFloat(-2800, 2800),
				RocketSim::Math::RandFloat(CommonValues::BALL_RADIUS, 650)
			);
			BallState ball = {};
			ball.pos = ballPos;
			ball.vel = Vec(
				RocketSim::Math::RandFloat(-900, 900),
				(RocketSim::Math::RandFloat() < 0.5f ? -1.f : 1.f) * RocketSim::Math::RandFloat(500, 2200),
				RocketSim::Math::RandFloat(-150, 250)
			);
			ball.angVel = Vec(
				RocketSim::Math::RandFloat(-3, 3),
				RocketSim::Math::RandFloat(-3, 3),
				RocketSim::Math::RandFloat(-3, 3)
			);
			arena->ball->SetState(ball);

			for (Car* car : arena->_cars) {
				const float teamDirection = car->team == Team::BLUE ? -1.f : 1.f;
				const float y = std::clamp(
					ballPos.y + teamDirection * RocketSim::Math::RandFloat(500, 1900),
					-3900.f,
					3900.f
				);
				CarState state = {};
				state.pos = Vec(
					std::clamp(ballPos.x + RocketSim::Math::RandFloat(-1400, 1400), -3200.f, 3200.f),
					y,
					17.f
				);
				state.vel = Vec(
					RocketSim::Math::RandFloat(-1100, 1100),
					RocketSim::Math::RandFloat(-1100, 1100),
					0.f
				);
				state.rotMat = Angle(RocketSim::Math::RandFloat(-3.14159265f, 3.14159265f), 0, 0).ToRotMat();
				state.boost = RocketSim::Math::RandFloat(15, 100);
				state.isOnGround = true;
				car->SetState(state);
			}
		}
	};

	class SetPlayState : public StateSetter {
	public:
		void ResetArena(Arena* arena) override {
			arena->ResetToRandomKickoff();

			const float side = RocketSim::Math::RandFloat() < 0.5f ? -1.f : 1.f;
			BallState ball = {};
			ball.pos = Vec(side * RocketSim::Math::RandFloat(700, 1500),
				RocketSim::Math::RandFloat(-450, 450), CommonValues::BALL_RADIUS);
			arena->ball->SetState(ball);

			int teamSlots[2] = { 0, 0 };
			for (Car* car : arena->_cars) {
				const int team = static_cast<int>(car->team);
				const int slot = teamSlots[team]++;
				const float teamDirection = car->team == Team::BLUE ? -1.f : 1.f;
				CarState state = {};
				state.pos = Vec((slot == 0 ? -650.f : 650.f) + side * 400.f,
					teamDirection * (slot == 0 ? 1450.f : 2200.f), 17.f);
				state.boost = RocketSim::Math::RandFloat(40, 100);
				state.isOnGround = true;
				car->SetState(state);
			}
		}
	};
}
