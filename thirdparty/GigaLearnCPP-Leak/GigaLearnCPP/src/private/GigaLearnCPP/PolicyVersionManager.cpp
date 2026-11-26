#include "PolicyVersionManager.h"
#include <nlohmann/json.hpp>

#include <GigaLearnCPP/Util/Utils.h>
#include "TrainingSchedule.h"

#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>

#include <private/GigaLearnCPP/PPO/PPOLearner.h>

#include <cmath>
#include <fstream>
#include <iomanip>

using namespace nlohmann;

GGL::PolicyVersionManager::PolicyVersionManager(
	std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion, 
	const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig, RenderSender* renderSender) : 
	saveFolder(saveFolder), maxVersions(maxVersions), tsPerVersion(tsPerVersion), 
	renderSender(renderSender) {

	skill.config = skillTrackerConfig;

	if (!std::filesystem::exists(saveFolder))
		std::filesystem::create_directories(saveFolder);

	if (skill.config.enabled) {
		RLGC::EnvSetConfig skillEnvSetConfig = envSetConfig;
		skillEnvSetConfig.numArenas = skill.config.numArenas;
		skill.envSet = new RLGC::EnvSet(skillEnvSetConfig);
		for (int i = 0; i < skill.envSet->arenas.size(); i++) {
			skill.envSet->rewards[i].clear();
			skill.envSet->stateSetters[i] = { new RLGC::FuzzedKickoffState() };
			skill.envSet->terminalConditions[i] = { new RLGC::GoalScoreCondition() };
		}
	} else {
		skill.envSet = NULL;
	}
}

GGL::PolicyVersion& GGL::PolicyVersionManager::AddVersion(ModelSet modelsToClone, uint64_t timesteps) {
	RG_NO_GRAD;

	auto models = modelsToClone.CloneAll();

	auto newVersion = PolicyVersion{
		timesteps,
		models
	};

	newVersion.ratings = skill.curRatings;

	versions.push_back(newVersion);

	SortVersions();

	// Remove old versions
	while (versions.size() > maxVersions) {
		auto& toRemove = versions[0];
		toRemove.models.Free();
		versions.erase(versions.begin());
	}

	return versions.back();
}

void GGL::PolicyVersionManager::SaveVersions() {
	RG_NO_GRAD;

	// Remove old saved versions
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(saveFolder);

	for (int64_t savedTimesteps : allSavedTimesteps) {
		bool matchesVersion = false;
		for (auto& version : versions)
			matchesVersion |= (savedTimesteps == version.timesteps);

		if (matchesVersion) {
			// We want to keep this
			allSavedTimesteps.insert(savedTimesteps);
		} else {
			// Get rid of it
			std::filesystem::remove_all(saveFolder / std::to_string(savedTimesteps));
		}
	}

	for (auto& version : versions) {
		if (allSavedTimesteps.contains(version.timesteps))
			continue;
		auto versionSaveFolder = saveFolder / std::to_string(version.timesteps);
		auto tempVersionSaveFolder = saveFolder / (std::to_string(version.timesteps) + ".tmp");
		if (std::filesystem::exists(tempVersionSaveFolder))
			std::filesystem::remove_all(tempVersionSaveFolder);
		std::filesystem::create_directories(tempVersionSaveFolder);

		version.models.Save(tempVersionSaveFolder, false);

		{ // Save JSON
			auto jsonPath = tempVersionSaveFolder / "STATS.json";

			std::ofstream fOut(jsonPath);
			RG_ASSERT(fOut.good());

			json j = {};
			j["skill_ratings"] = version.ratings.ToJSON();
			std::string jStr = j.dump(4);
			fOut << jStr;
		}

		std::filesystem::rename(tempVersionSaveFolder, versionSaveFolder);
	}
}

void GGL::PolicyVersionManager::LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps) {

	RG_NO_GRAD;

	RG_LOG("PolicyVersionManager::LoadVersions():");

	for (auto& version : versions)
		version.models.Free();
	versions.clear();

	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(saveFolder);

	for (int64_t savedTimesteps : allSavedTimesteps) {

		if (savedTimesteps > curTimesteps) {
			RG_ERR_CLOSE(
				"Tried to load saved policy version that is newer than our current model (" << savedTimesteps << " > " << curTimesteps << ")!\n" <<
				"If you deleted some checkpoints, make sure to delete that far back in the saved policy versions as well");
		}
		auto path = saveFolder / std::to_string(savedTimesteps);
		PolicyVersion& version = AddVersion(modelsTemplate, savedTimesteps);
		version.models.Load(path, false, false);

		{ // Load JSON
			// TODO: Repetitive
			auto jsonPath = path / "STATS.json";
			std::ifstream fIn(jsonPath);
			RG_ASSERT(fIn.good());

			json j = json::parse(fIn);
			if (j.contains("skill_ratings"))
				version.ratings.ReadFromJSON(j["skill_ratings"]);
		}
	}

	SortVersions();

	RG_LOG(" > Loaded " << versions.size() << " versions(s)");
}

void GGL::PolicyVersionManager::SortVersions() {
	auto fnCompareVersions = [](const PolicyVersion& a, const PolicyVersion& b) {
		return a.timesteps < b.timesteps;
	};

	std::sort(versions.begin(), versions.end(), fnCompareVersions);
}

/////////////////////////////////////////////////////////////////////

void GGL::PolicyVersionManager::RunSkillMatches(PPOLearner* ppo, Report& report, int64_t totalTimesteps) {
	RG_NO_GRAD;
	
	auto fnUpdateRatings = [this](SkillRating& winner, SkillRating& loser, RLGC::GameState& state) {
		float& winnerRating = winner.GetRating(state, skill.config.initialRating);
		float& loserRating = loser.GetRating(state, skill.config.initialRating);
		
		// Update according to ELO math
		float expDelta = (loserRating - winnerRating) / 400;
		float expected = 1 / (powf(10, expDelta) + 1);

		winnerRating += skill.config.ratingInc * (1 - expected);
		loserRating += skill.config.ratingInc * (expected - 1);
	};

	
	Team newTeam;
	int oldVersionIndex;
	float totalSimTime;
	if (skill.doContinuation) {
		RG_ASSERT(skill.prevOldVersionIndex < versions.size());
		oldVersionIndex = skill.prevOldVersionIndex;
		newTeam = skill.prevNewTeam;
		totalSimTime = skill.prevSimTime;
	} else {
		std::vector<int> olderVersionIndices;
		for (int i = 0; i < versions.size(); i++) {
			if (versions[i].timesteps < static_cast<uint64_t>(totalTimesteps))
				olderVersionIndices.push_back(i);
		}
		if (olderVersionIndices.empty()) {
			RG_LOG("Skipping checkpoint ELO: no older policy version is available.");
			return;
		}
		oldVersionIndex = olderVersionIndices[Math::RandInt(0, olderVersionIndices.size())];
		newTeam = (Team)Math::RandInt(0, 2);
		totalSimTime = 0;

		skill.envSet->Reset();
	}
	skill.doContinuation = false;

	auto& oldVersion = versions[oldVersionIndex];

	// Find which players are on which teams
	std::vector<int>
		newPlayers = {}, 
		oldPlayers = {};
	for (int i = 0; i < skill.envSet->arenas.size(); i++) {
		auto& state = skill.envSet->state.gameStates[i];
		for (int j = 0; j < state.players.size(); j++) {
			int playerIdx = skill.envSet->state.arenaPlayerStartIdx[i] + j;
			bool isNew = (state.players[j].team == newTeam);
			(isNew ? newPlayers : oldPlayers).push_back(playerIdx);
		}
	}

	torch::Tensor
		tNewPlayers = torch::tensor(newPlayers),
		tOldPlayers = torch::tensor(oldPlayers);

	int newGoals = 0, oldGoals = 0;

	RG_LOG("Running skill matches (simTime=" << skill.config.simTime << ")...");

	SkillRating prevCurRatings = skill.curRatings;

	float stepTime = skill.envSet->config.tickSkip * RLGC::CommonValues::TICK_TIME;
	for (float t = 0; 
		t < skill.config.simTime && totalSimTime < skill.config.maxSimTime && skill.curGoals < skill.envSet->arenas.size();
		t += stepTime, totalSimTime += stepTime) {

		skill.envSet->Reset();

		torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(skill.envSet->state.obs);
		torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(skill.envSet->state.actionMasks);

		torch::Tensor tNewStates = tStates.index_select(0, tNewPlayers);
		torch::Tensor tOldStates = tStates.index_select(0, tOldPlayers);

		torch::Tensor tNewActionMasks = tActionMasks.index_select(0, tNewPlayers);
		torch::Tensor tOldActionMasks = tActionMasks.index_select(0, tOldPlayers);

		skill.envSet->StepFirstHalf(true);

		torch::Tensor tNewActions, tOldActions;
		torch::Tensor _tLogProbs;

		PPOLearner::InferActionsFromModels(
			ppo->models, tNewStates.to(ppo->device, true), tNewActionMasks.to(ppo->device, true), 
			skill.config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision, 
			&tNewActions, &_tLogProbs);
		PPOLearner::InferActionsFromModels(
			oldVersion.models, tOldStates.to(ppo->device, true), tOldActionMasks.to(ppo->device, true), 
			skill.config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
			&tOldActions, &_tLogProbs);

		auto newActions = TENSOR_TO_VEC<int>(tNewActions);
		auto oldActions = TENSOR_TO_VEC<int>(tOldActions);

		auto combinedActions = std::vector<int>(skill.envSet->state.numPlayers, -1);
		for (int i = 0; i < newActions.size(); i++)
			combinedActions[newPlayers[i]] = newActions[i];
		for (int i = 0; i < oldActions.size(); i++)
			combinedActions[oldPlayers[i]] = oldActions[i];

		skill.envSet->Sync();
		skill.envSet->StepSecondHalf(combinedActions, false);

		for (int i = 0; i < skill.envSet->arenas.size(); i++) {
			auto& gs = skill.envSet->state.gameStates[i];
			if (gs.goalScored) {
				std::string modeName = SkillRating::GetModeName(gs);

				if (RS_TEAM_FROM_Y(gs.ball.pos.y) != newTeam) {
					fnUpdateRatings(skill.curRatings, oldVersion.ratings, gs);
				} else {
					fnUpdateRatings(oldVersion.ratings, skill.curRatings, gs);
				}

				skill.curGoals++;
			}
		}

		if (renderSender)
			renderSender->Send(skill.envSet->state.gameStates[0]);
	}

	const std::string modeName = SkillRating::GetModeName(skill.envSet->state.gameStates[0]);
	const float currentRating = skill.curRatings.GetRating(modeName, skill.config.initialRating);
	const float previousRating = prevCurRatings.GetRating(modeName, skill.config.initialRating);
	RG_LOG("Checkpoint ELO @ " << totalTimesteps << " timesteps (" << modeName << "): "
		<< currentRating << " (" << (currentRating - previousRating >= 0 ? '+' : '-')
		<< std::abs(currentRating - previousRating) << ")");
	report["Rating/" + modeName] = currentRating;

	for (auto& version : versions) {
		if (version.timesteps == static_cast<uint64_t>(totalTimesteps)) {
			version.ratings = skill.curRatings;
			break;
		}
	}

	const auto eloPath = saveFolder.parent_path() / "elo_history.csv";
	const bool writeHeader = !std::filesystem::exists(eloPath) || std::filesystem::file_size(eloPath) == 0;
	std::ofstream eloOut(eloPath, std::ios::app);
	if (writeHeader)
		eloOut << "total_timesteps,mode,elo\n";
	eloOut << totalTimesteps << ',' << modeName << ',' << std::setprecision(8) << currentRating << '\n';

	if (skill.curGoals < skill.envSet->arenas.size() && totalSimTime < skill.config.maxSimTime) {
		// Not enough goals were scored, we will force a continuation where the same models keep playing from the end position
		// This COULD switch versions, but it wouldn't be a big deal as it would just be the immediate next version
		RG_LOG(" > Forcing continuation (" << skill.curGoals <<  "/" << skill.envSet->arenas.size() << ")");
		skill.doContinuation = true;
		skill.prevOldVersionIndex = oldVersionIndex;
		skill.prevNewTeam = newTeam;
		skill.prevSimTime = totalSimTime;
	} else {
		skill.curGoals = 0;
	}
}

void GGL::PolicyVersionManager::OnIteration(struct PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps) {
	const bool savedNewVersion = Carnage::Training::CrossedCheckpoint(
		static_cast<uint64_t>(prevTotalTimesteps),
		static_cast<uint64_t>(totalTimesteps),
		static_cast<uint64_t>(tsPerVersion)
	);
	if (savedNewVersion) {
		// Save version
		AddVersion(ppo->GetPolicyModels(), totalTimesteps);
	}

	if (skill.config.enabled) {
		if (skill.config.updateOnNewVersion) {
			if (savedNewVersion && versions.size() > 1)
				RunSkillMatches(ppo, report, totalTimesteps);
		} else {
			skill.iterationsSinceRan++;
			if (skill.iterationsSinceRan >= skill.config.updateInterval && !versions.empty()) {
				skill.iterationsSinceRan = 0;
				RunSkillMatches(ppo, report, totalTimesteps);
			}
		}
	}
}

void GGL::PolicyVersionManager::AddRunningStatsToJSON(nlohmann::json& json) {
	if (skill.config.enabled) {
		json["skill_ratings"] = skill.curRatings.ToJSON();
		nlohmann::json versionRatings = nlohmann::json::object();
		for (const auto& version : versions)
			versionRatings[std::to_string(version.timesteps)] = version.ratings.ToJSON();
		json["policy_version_ratings"] = std::move(versionRatings);
	}
}

void GGL::PolicyVersionManager::LoadRunningStatsFromJSON(const nlohmann::json& json) {
	if (skill.config.enabled) {
		if (json.contains("skill_ratings"))
			skill.curRatings.ReadFromJSON(json["skill_ratings"]);
		if (json.contains("policy_version_ratings") && json["policy_version_ratings"].is_object())
			skill.pendingVersionRatings = json["policy_version_ratings"];
		else
			skill.pendingVersionRatings = nlohmann::json::object();
	}
}

bool GGL::PolicyVersionManager::HasPendingVersionRating(uint64_t timesteps) const {
	return skill.pendingVersionRatings.is_object()
		&& skill.pendingVersionRatings.contains(std::to_string(timesteps));
}

void GGL::PolicyVersionManager::ApplyPendingVersionRatings() {
	if (!skill.config.enabled || !skill.pendingVersionRatings.is_object())
		return;

	for (auto& version : versions) {
		const auto key = std::to_string(version.timesteps);
		if (skill.pendingVersionRatings.contains(key))
			version.ratings.ReadFromJSON(skill.pendingVersionRatings[key]);
	}
	skill.pendingVersionRatings = nlohmann::json::object();
}
