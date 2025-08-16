Phase 1: Environment Setup & Integration
[x] Create Project Structure

[x] Initialize the main project directory.

[x] Create a framework subdirectory and move your existing backtesting and validation scripts into it.

[x] Create an rl_agent subdirectory for all new code.

[x] Install Dependencies

[x] Run pip install gymnasium stable-baselines3 pandas-ta numpy pandas.

[ ] Build the Environment Wrapper (StrategyEnv.py)

[ ] Create the StrategyEnv.py file with a class StrategyEnv that inherits from gymnasium.Env.

[ ] Implement the __init__ method to load your historical data and define the action_space (e.g., Discrete(20) for 20 possible actions) and observation_space (e.g., Dict to describe the current strategy).

Phase 2: Defining the RL Logic & Core Components
[ ] Implement the step(action) Method

[ ] Translate the agent's action (an integer) into a specific modification of the current trading strategy.

[ ] When the "run_backtest" action is triggered, call your existing framework functions (dameSistema, calculaCurvas, etc.).

[ ] Return the standard (observation, reward, terminated, truncated, info) tuple.

[ ] Design the _calculate_reward() Function

[ ] Create a specific, non-trivial reward function that balances risk and return.

[ ] Implement a formula (e.g., reward = (CAGR / Max_Drawdown)) inside the environment.

[ ] Add heavy negative penalties for bad outcomes (e.g., drawdown > 30%, fewer than 10 trades).

[ ] Implement the reset() Method

[ ] Code the reset() function to clear the current strategy and prepare for a new episode.

[ ] Ensure it returns the initial observation.

Phase 3: Training, Stress-Testing & Final Evaluation
[ ] Build the GARCH Stress-Test Module (garch_tester.py)

[ ] Create a standalone function run_garch_test(strategy_definition).

[ ] This function will fit a GARCH model to real data, generate 100+ synthetic price series, and backtest the provided strategy on each one.

[ ] The function should return a final robustness score (e.g., the percentage of profitable simulations).

[ ] Create the Training Script (train.py)

[ ] Instantiate your StrategyEnv.

[ ] Choose and initialize a model from stable-baselines3 (e.g., model = PPO("MlpPolicy", env)).

[ ] Run the training loop with model.learn(total_timesteps=100000).

[ ] Save the trained agent's weights using model.save().

[ ] Create the Final Evaluation Script (evaluate.py)

[ ] Load the saved agent.

[ ] Use the agent to predict its single, best-found strategy.

[ ] Execute the Validation Gauntlet:

[ ] Stage 1: Test the strategy on your real, unseen Out-of-Sample (OS) data.

[ ] Stage 2: If Stage 1 is successful, run the strategy through the GARCH Stress-Test module.

[ ] Print a final report detailing the strategy's rules and its performance across all validation stages.****