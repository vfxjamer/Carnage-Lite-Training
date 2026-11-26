#pragma once

#include <RLGymCPP/ActionParsers/DefaultAction.h>

namespace RLGC {
	// Repeats jump-bearing controls in the discrete table, increasing their
	// sampling probability while preserving DefaultAction's legality masks.
	class CarnageAction : public DefaultAction {
	public:
		CarnageAction() : DefaultAction() {
			const auto originalActions = actions;
			const auto originalGroundMask = groundMask;
			const auto originalAirMask = airMask;
			const auto originalJumpMask = jumpMask;
			const auto originalBoostMask = boostMask;

			for (size_t i = 0; i < originalActions.size(); i++) {
				if (!originalActions[i].jump)
					continue;

				actions.push_back(originalActions[i]);
				groundMask.push_back(originalGroundMask[i]);
				airMask.push_back(originalAirMask[i]);
				jumpMask.push_back(originalJumpMask[i]);
				boostMask.push_back(originalBoostMask[i]);
			}
		}
	};
}
