#include <GigaLearnCPP/Learner.h>
#include "TrainingSchedule.h"
#include <nlohmann/json.hpp>

#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include "CarnageAction.h"
#include "CarnageRewards.h"
#include "CarnageStateSetters.h"

using namespace GGL;
using namespace RLGC;

namespace {
	struct PhaseSettings {
		int64_t timestepsPerIteration;
		int64_t miniBatchSize;
		float policyLR;
		float gamma;
	};

	struct TrainingRuntime {
		Carnage::Training::GameMode mode = Carnage::Training::GameMode::OneVOne;
		uint64_t modeStartTimesteps = 0;
		uint64_t countedTimesteps = 0;
		int phase = 1;
		std::filesystem::path rewardMetricsPath;
	} runtime;

	const char* ModeName(Carnage::Training::GameMode mode) {
		return mode == Carnage::Training::GameMode::TwoVTwo ? "2v2" : "1v1";
	}

	PhaseSettings GetPhaseSettings(int phase) {
		switch (phase) {
		case 1: return { 50'000, 25'000, 2e-4f, 0.99f };
		case 2: return { 100'000, 25'000, 1e-4f, 0.99f };
		case 3: return { 150'000, 50'000, 1e-4f, 0.995f };
		case 4: return { 300'000, 50'000, 0.8e-4f, 0.998f };
		case 5: return { 400'000, 50'000, 0.8e-4f, 0.998f };
		default: throw std::invalid_argument("Phase must be between 1 and 5");
		}
	}

	std::vector<WeightedReward> MakeRewards(int phase, float teamSpirit) {
		std::vector<WeightedReward> result;
		const auto earlyWeights = Carnage::Training::GetEarlyPhaseRewardWeights(phase);
		auto add = [&](const char* name, Reward* reward, float weight, bool zeroSum = false) {
			if (zeroSum)
				reward = new ZeroSumReward(reward, teamSpirit);
			result.emplace_back(new EpisodeMetricReward(name, reward, weight), weight);
		};

		if (phase == 1) {
			add("TouchBall", new TouchBallReward(), earlyWeights.touchBall);
			add("SpeedTowardBall", new SpeedTowardBallReward(), earlyWeights.speedTowardBall, true);
			add("FaceBall", new FaceBallReward(), earlyWeights.faceBall);
			add("InAir", new AirReward(), earlyWeights.inAir);
			return result;
		}

		if (phase == 2) {
			add("TouchBall", new TouchBallReward(), earlyWeights.touchBall);
			add("SpeedTowardBall", new SpeedTowardBallReward(), earlyWeights.speedTowardBall, true);
			add("FaceBall", new FaceBallReward(), earlyWeights.faceBall);
			add("InAir", new AirReward(), earlyWeights.inAir);
			add("BallVelocityTowardGoal", new VelocityBallToGoalReward(), earlyWeights.ballVelocityTowardGoal);
			add("GoalAndConcede", new GoalReward(earlyWeights.concedeScale), earlyWeights.goalWeight);
			return result;
		}

		const bool phase3 = phase == 3;
		const bool phase4 = phase == 4;
		add("StrongTouch", new StrongTouchReward(), phase3 ? 15.f : phase4 ? 10.f : 8.f, true);
		add("SpeedTowardBall", new SpeedTowardBallReward(), phase3 ? 0.3f : 0.2f, true);
		add("FaceBall", new FaceBallReward(), phase3 ? 0.03f : 0.02f);
		add("InAir", new AirReward(), phase3 ? 0.02f : 0.01f);
		add("BallVelocityTowardGoal", new VelocityBallToGoalReward(), 1.5f, phase4 || phase == 5);
		add("GoalAndConcede", new GoalReward(-0.8f), 60.f);
		add("SaveBoost", new SaveBoostReward(0.5f), 0.2f);
		add("BoostPickup", new PickupBoostReward(), 3.f, true);
		add("AirTouch", new AirTouchReward(), phase3 || phase4 ? 25.f : 15.f);
		add("DemoOrBump", new DemoBumpReward(), 10.f, true);
		return result;
	}

	StateSetter* MakeStateSetter(int phase) {
		auto random = [] { return new RandomState(true, true, false); };
		auto kickoff = [] { return new KickoffState(); };
		switch (phase) {
		case 1:
			return random();
		case 2:
			return new CombinedState({ { random(), 75.f }, { kickoff(), 25.f } });
		case 3:
			return new CombinedState({ { random(), 50.f }, { kickoff(), 50.f } });
		case 4:
			return new CombinedState({
				{ kickoff(), 70.f },
				{ random(), 20.f },
				{ new SetPlayState(), 10.f }
			});
		case 5:
			return new CombinedState({
				{ kickoff(), 50.f },
				{ new ProceduralMidGameState(), 50.f }
			});
		default:
			throw std::invalid_argument("Phase must be between 1 and 5");
		}
	}

	uint64_t FindLatestCheckpointStep(const std::filesystem::path& checkpointFolder) {
		uint64_t latest = 0;
		if (!std::filesystem::exists(checkpointFolder))
			return latest;

		for (const auto& entry : std::filesystem::directory_iterator(checkpointFolder)) {
			if (!entry.is_directory())
				continue;
			const auto name = entry.path().filename().string();
			if (!Carnage::Training::IsNumericCheckpointName(name))
				continue;
			try {
				latest = std::max(latest, static_cast<uint64_t>(std::stoull(name)));
			} catch (...) {
				// Non-numeric subdirectories, such as policy_versions, are not checkpoints.
			}
		}
		return latest;
	}

	uint64_t ResolveTwoVTwoStartStep(
		const std::filesystem::path& checkpointFolder,
		uint64_t latestCheckpoint,
		int64_t requestedStartStep,
		bool automaticMode
	) {
		const auto modeStatePath = checkpointFolder / "mode_state.json";
		const auto modeStateTempPath = checkpointFolder / "mode_state.json.tmp";
		const auto modeStateBackupPath = checkpointFolder / "mode_state.json.bak";
		uint64_t startStep = requestedStartStep >= 0
			? static_cast<uint64_t>(requestedStartStep)
			: latestCheckpoint;
		bool loadedPersistedStart = false;

		if (requestedStartStep < 0) {
			const std::array<std::filesystem::path, 2> modeStateCandidates = {
				modeStatePath,
				modeStateBackupPath
			};
			for (const auto& candidate : modeStateCandidates) {
				if (!std::filesystem::exists(candidate))
					continue;
				try {
					std::ifstream input(candidate);
					nlohmann::json state;
					input >> state;
					if (!state.contains("two_v_two_start_step"))
						continue;
					const uint64_t savedStart = state["two_v_two_start_step"].get<uint64_t>();
					const bool validBoundary = !automaticMode || savedStart >= Carnage::Training::kOneVOneTimesteps;
					if (savedStart <= latestCheckpoint && validBoundary) {
						startStep = savedStart;
						loadedPersistedStart = true;
						break;
					}
				} catch (...) {
					RG_LOG("Could not read " << candidate.filename().string()
						<< "; trying the other mode-state copy.");
				}
			}
		}

		if (automaticMode && startStep < Carnage::Training::kOneVOneTimesteps)
			startStep = latestCheckpoint;
		if (loadedPersistedStart)
			return startStep;

		nlohmann::json state = {
			{"schema_version", 1},
			{"two_v_two_start_step", startStep}
		};
		try {
			std::filesystem::create_directories(checkpointFolder);
			if (std::filesystem::exists(modeStateTempPath))
				std::filesystem::remove(modeStateTempPath);
			{
				std::ofstream output(modeStateTempPath, std::ios::trunc);
				if (!output.good())
					throw std::runtime_error("cannot open temporary mode-state file");
				output << state.dump(2) << '\n';
				output.flush();
				if (!output.good())
					throw std::runtime_error("failed to flush temporary mode-state file");
			}

			bool movedPreviousState = false;
			if (std::filesystem::exists(modeStatePath)) {
				if (std::filesystem::exists(modeStateBackupPath))
					std::filesystem::remove(modeStateBackupPath);
				std::filesystem::rename(modeStatePath, modeStateBackupPath);
				movedPreviousState = true;
			}
			try {
				std::filesystem::rename(modeStateTempPath, modeStatePath);
				if (std::filesystem::exists(modeStateBackupPath))
					std::filesystem::remove(modeStateBackupPath);
			} catch (...) {
				if (movedPreviousState && !std::filesystem::exists(modeStatePath)
					&& std::filesystem::exists(modeStateBackupPath))
					std::filesystem::rename(modeStateBackupPath, modeStatePath);
				throw;
			}
		} catch (const std::exception& e) {
			throw std::runtime_error(std::string("Could not persist mode_state.json safely: ") + e.what());
		}
		return startStep;
	}

	void UpdateTeamSpirit(Learner* learner, float teamSpirit) {
		for (auto& arenaRewards : learner->envSet->rewards) {
			for (auto& weighted : arenaRewards) {
				auto* metricsReward = dynamic_cast<EpisodeMetricReward*>(weighted.reward);
				if (!metricsReward)
					continue;
				auto* zeroSum = dynamic_cast<ZeroSumReward*>(metricsReward->Child());
				if (zeroSum)
					zeroSum->teamSpirit = teamSpirit;
			}
		}
	}

	void AppendRewardEpisodeMetrics(
		Learner* learner,
		const GameState&,
		size_t arenaIndex,
		Report& report
	) {
		if (learner->envSet->state.terminals[arenaIndex] == TerminalType::NOT_TERMINAL)
			return;

		struct Row {
			std::string name;
			int teamIndex;
			Carnage::Metrics::TeamEpisodeRewardStats stats;
		};
		std::vector<Row> rows;
		double totalAbsoluteReward = 0;
		for (auto& weighted : learner->envSet->rewards[arenaIndex]) {
			auto* metric = dynamic_cast<EpisodeMetricReward*>(weighted.reward);
			if (!metric)
				continue;
			auto episodeStats = metric->TakeEpisodeStats();
			for (int teamIndex = 0; teamIndex < 2; teamIndex++) {
				const auto& stats = episodeStats.teams[teamIndex];
				if (!stats.sampleCount)
					continue;
				rows.push_back({ metric->GetName(), teamIndex, stats });
				totalAbsoluteReward += stats.weightedAbsoluteTotal;
			}
		}

		if (rows.empty())
			return;

		const auto path = runtime.rewardMetricsPath;
		std::filesystem::create_directories(path.parent_path());
		const bool writeHeader = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
		std::ofstream output(path, std::ios::app);
		if (writeHeader)
			output << "total_timesteps,mode,phase,team,reward,mean_per_player_step,episode_total_per_player,share_absolute_reward\n";
		output << std::fixed << std::setprecision(6);

		for (const auto& row : rows) {
			const char* teamName = row.teamIndex == 0 ? "Blue" : "Orange";
			const double share = totalAbsoluteReward > 0
				? row.stats.weightedAbsoluteTotal / totalAbsoluteReward
				: 0;
			const double meanPerStep = row.stats.MeanPerPlayerStep();
			const double episodeTotal = row.stats.EpisodeTotalPerPlayer();
			const std::string prefix = "Rewards/" + row.name + "/" + teamName + "/";
			report.AddAvg(prefix + "MeanPerPlayerStep", meanPerStep);
			report.AddAvg(prefix + "EpisodeTotalPerPlayer", episodeTotal);
			report.AddAvg(prefix + "ShareAbsoluteReward", share);
			output << runtime.countedTimesteps << ',' << ModeName(runtime.mode) << ',' << runtime.phase << ','
				<< teamName << ',' << row.name << ',' << meanPerStep << ',' << episodeTotal << ',' << share << '\n';
		}
	}

	void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
		uint64_t collectedTimesteps = 0;
		for (size_t i = 0; i < states.size(); i++) {
			collectedTimesteps += states[i].players.size();
			AppendRewardEpisodeMetrics(learner, states[i], i, report);
			if (states[i].goalScored) {
				report.AddAvg("Game/Goal Scored", 1.f);
				report.AddAvg("Game/Goal Speed", states[i].ball.vel.Length());
			}
		}

		runtime.countedTimesteps += collectedTimesteps;
		if (runtime.mode == Carnage::Training::GameMode::TwoVTwo) {
			const float teamSpirit = Carnage::Training::TeamSpiritAt(
				runtime.countedTimesteps, runtime.modeStartTimesteps);
			UpdateTeamSpirit(learner, teamSpirit);
			report.AddAvg("Training/TeamSpirit", teamSpirit);
		}
	}
}

int main(int argc, char* argv[]) {
	std::string meshesPath = "collision_meshes";
	std::string deviceStr = "cpu";
	std::string requestedMode = "auto";
	int phase = 1;
	int numGames = 256;
	int64_t maxTotalTimesteps = 0;
	int64_t maxIterations = 0;
	int64_t modeStartTimesteps = -1;
	float policyLROverride = 0;
	std::string saveDir = "checkpoints";

	auto nextArg = [&](int& index, const char* flag) -> std::string {
		if (index + 1 >= argc)
			throw std::invalid_argument(std::string("Missing value for ") + flag);
		return argv[++index];
	};

	try {
		for (int i = 1; i < argc; i++) {
			const std::string arg = argv[i];
			if (arg == "--device") deviceStr = nextArg(i, "--device");
			else if (arg == "--mode") requestedMode = nextArg(i, "--mode");
			else if (arg == "--phase") phase = std::stoi(nextArg(i, "--phase"));
			else if (arg == "--games") numGames = std::stoi(nextArg(i, "--games"));
			else if (arg == "--save-dir") saveDir = nextArg(i, "--save-dir");
			else if (arg == "--max-steps") maxTotalTimesteps = std::stoll(nextArg(i, "--max-steps"));
			else if (arg == "--max-iterations") maxIterations = std::stoll(nextArg(i, "--max-iterations"));
			else if (arg == "--mode-start-step") modeStartTimesteps = std::stoll(nextArg(i, "--mode-start-step"));
			else if (arg == "--policy-lr") policyLROverride = std::stof(nextArg(i, "--policy-lr"));
			else if (!arg.starts_with("--")) meshesPath = arg;
			else throw std::invalid_argument("Unknown option: " + arg);
		}
	} catch (const std::exception& e) {
		RG_LOG("Invalid command line: " << e.what());
		return EXIT_FAILURE;
	}

	if (!Carnage::Training::IsEnabledPhase(phase) || numGames <= 0) {
		RG_LOG("For this training run, --phase must be 1 or 2 and --games must be positive");
		return EXIT_FAILURE;
	}
	if (deviceStr != "cpu" && deviceStr != "cuda" && deviceStr != "auto") {
		RG_LOG("--device must be cpu, cuda, or auto");
		return EXIT_FAILURE;
	}
	if (requestedMode != "auto" && requestedMode != "1v1") {
		RG_LOG("For this training run, --mode must be auto or 1v1");
		return EXIT_FAILURE;
	}

	const uint64_t latestCheckpoint = FindLatestCheckpointStep(saveDir);
	runtime.mode = requestedMode == "auto"
		? Carnage::Training::ModeAt(latestCheckpoint)
		: Carnage::Training::GameMode::OneVOne;
	if (runtime.mode != Carnage::Training::GameMode::OneVOne) {
		RG_LOG("2v2 is disabled in this P1/P2 training build");
		return EXIT_FAILURE;
	}
	runtime.phase = phase;
	try {
		runtime.modeStartTimesteps = runtime.mode == Carnage::Training::GameMode::TwoVTwo
			? ResolveTwoVTwoStartStep(saveDir, latestCheckpoint, modeStartTimesteps, requestedMode == "auto")
			: 0;
	} catch (const std::exception& e) {
		RG_LOG("Cannot safely start 2v2 training: " << e.what());
		return EXIT_FAILURE;
	}
	runtime.countedTimesteps = latestCheckpoint;
	runtime.rewardMetricsPath = std::filesystem::path(saveDir) / "reward_episode_metrics.csv";

	const float initialTeamSpirit = runtime.mode == Carnage::Training::GameMode::TwoVTwo
		? Carnage::Training::TeamSpiritAt(latestCheckpoint, runtime.modeStartTimesteps)
		: 0.f;
	const int playersPerTeam = runtime.mode == Carnage::Training::GameMode::TwoVTwo ? 2 : 1;

	RG_LOG("Carnage-Lite (8.9M)");
	RG_LOG("Mode: " << ModeName(runtime.mode) << " | phase: " << phase
		<< " | checkpoint step: " << latestCheckpoint
		<< " | initial team spirit: " << initialTeamSpirit);

	RocketSim::Init(meshesPath);

	auto envCreateFunc = [phase, playersPerTeam, initialTeamSpirit](int) -> EnvCreateResult {
		EnvCreateResult result = {};
		result.arena = Arena::Create(GameMode::SOCCAR);
		for (int i = 0; i < playersPerTeam; i++) {
			result.arena->AddCar(Team::BLUE);
			result.arena->AddCar(Team::ORANGE);
		}
		result.actionParser = new CarnageAction();
		result.obsBuilder = new DefaultObsPadded(2);
		result.stateSetter = MakeStateSetter(phase);
		result.terminalConditions = {
			new NoTouchCondition(10),
			new GoalScoreCondition()
		};
		result.rewards = MakeRewards(phase, initialTeamSpirit);
		return result;
	};

	LearnerConfig config = {};
	config.deviceType = deviceStr == "cpu" ? LearnerDeviceType::CPU
		: deviceStr == "cuda" ? LearnerDeviceType::GPU_CUDA : LearnerDeviceType::AUTO;
	config.renderMode = false;
	config.tickSkip = 8;
	config.actionDelay = config.tickSkip - 1;
	config.numGames = numGames;
	config.checkpointFolder = saveDir;
	config.checkpointsToKeep = 8;
	config.tsPerSave = Carnage::Training::kCheckpointTimesteps;
	config.tsPerVersion = Carnage::Training::kCheckpointTimesteps;
	config.maxOldVersions = 32;
	config.maxTotalTimesteps = static_cast<int64_t>(
		Carnage::Training::BoundedRunTimesteps(maxTotalTimesteps));
	config.maxIterations = maxIterations;
	config.randomSeed = 123;
	config.standardizeReturns = true;
	config.addRewardsToMetrics = true;
	config.sendMetrics = false;
	config.savePolicyVersions = true;
	config.skillTracker.enabled = true;
	config.skillTracker.numArenas = 16;
	config.skillTracker.simTime = 45;
	config.skillTracker.maxSimTime = 240;
	config.skillTracker.ratingInc = 5;
	config.skillTracker.initialRating = 0;
	config.skillTracker.deterministic = true;
	config.skillTracker.updateOnNewVersion = true;

	const auto phaseSettings = GetPhaseSettings(phase);
	config.ppo.tsPerItr = phaseSettings.timestepsPerIteration;
	config.ppo.batchSize = phaseSettings.timestepsPerIteration;
	config.ppo.miniBatchSize = phaseSettings.miniBatchSize;
	config.ppo.epochs = 2;
	config.ppo.entropyScale = 0.01f;
	config.ppo.clipRange = 0.2f;
	config.ppo.gaeGamma = phaseSettings.gamma;
	config.ppo.gaeLambda = 0.95f;
	config.ppo.policyLR = policyLROverride > 0 ? policyLROverride : phaseSettings.policyLR;
	config.ppo.criticLR = config.ppo.policyLR;
	config.ppo.sharedHead = {};
	config.ppo.policy.layerSizes = { 2048, 2048, 1024 };
	config.ppo.critic.layerSizes = { 2048, 2048, 1024 };
	config.ppo.policy.addLayerNorm = true;
	config.ppo.critic.addLayerNorm = true;
	config.ppo.policy.activationType = ModelActivationType::RELU;
	config.ppo.critic.activationType = ModelActivationType::RELU;
	config.ppo.policy.optimType = ModelOptimType::ADAM;
	config.ppo.critic.optimType = ModelOptimType::ADAM;

	Learner learner(envCreateFunc, config, StepCallback);
	runtime.countedTimesteps = learner.totalTimesteps;
	const uint64_t parameterCount = learner.GetModelParameterCount();
	RG_LOG("Observation size: " << learner.obsSize << " | action count: " << learner.numActions);
	RG_LOG("Policy + critic parameters: " << parameterCount);
	RG_LOG("Checkpoint ELO: every 5,000,000 timesteps; ELO only, separated by game mode.");
	RG_LOG("Reward episode details: " << runtime.rewardMetricsPath.string());

	learner.Start();
	return EXIT_SUCCESS;
}
