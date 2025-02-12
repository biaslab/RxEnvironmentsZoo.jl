@testitem "StochasticMaze" begin
    using RxEnvironmentsZoo
    using Distributions
    using LinearAlgebra
    using RxEnvironments
    # Test environment creation with valid inputs
    @testset "Environment Creation" begin
        # Create a simple 2-state environment with 2 actions
        transition_tensor = zeros(2, 2, 2)
        # Action 1
        transition_tensor[:, 1, 1] = [0.8, 0.2]  # From state 1
        transition_tensor[:, 2, 1] = [0.3, 0.7]  # From state 2
        # Action 2
        transition_tensor[:, 1, 2] = [0.6, 0.4]  # From state 1
        transition_tensor[:, 2, 2] = [0.1, 0.9]  # From state 2

        observation_matrix = [0.9 0.1; 0.1 0.9]  # 2 observation states
        reward_states = [(2, 1.0)]  # State 2 has reward 1.0

        # Should create without error
        @test_nowarn StochasticMaze(transition_tensor, observation_matrix, reward_states)

        # Test invalid transition probabilities
        invalid_transition = copy(transition_tensor)
        invalid_transition[:, 1, 1] = [0.7, 0.7]  # Sum > 1
        @test_throws ArgumentError StochasticMaze(invalid_transition, observation_matrix, reward_states)

        # Test invalid observation matrix
        invalid_obs = copy(observation_matrix)
        invalid_obs[:, 1] = [-0.1, 1.1]  # Invalid probabilities
        @test_throws ArgumentError StochasticMaze(transition_tensor, invalid_obs, reward_states)


        # Test environment creation with 5 states and 3 actions
        n_states = 20
        n_actions = 3
        n_observations = 20
        transition_tensor = zeros(n_states, n_states, n_actions)

        # Set up transition probabilities for each action with skewed distributions
        for a in 1:n_actions
            for s in 1:n_states
                # Create biased Dirichlet parameters - higher alpha for staying in same state
                # and transitioning to adjacent states
                alphas = ones(n_states) * 0.5  # Base concentration of 0.5
                preferred_state = rand(1:n_states)
                alphas[preferred_state] = 3.0  # Bias towards going to preferred state
                adjacent_states = [mod1(preferred_state - 1, n_states), mod1(preferred_state + 1, n_states)]
                for as in adjacent_states
                    alphas[as] = 1.5
                end

                # Sample from Dirichlet distribution
                transition_tensor[:, s, a] = rand(Dirichlet(alphas))
            end
        end

        # Create observation matrix with 3 possible observations
        observation_matrix = zeros(n_observations, n_states)
        for s in 1:n_states
            probs = rand(n_observations)
            observation_matrix[:, s] = probs / sum(probs)
        end

        reward_states = [(3, 1.0), (5, -1.0)] # Rewards in states 3 and 5

        # Should create without error
        @test_nowarn StochasticMaze(transition_tensor, observation_matrix, reward_states)
    end

    # Test state transitions and observations
    @testset "State Transitions and Observations" begin
        # Create a deterministic transition for testing
        transition_tensor = zeros(2, 2, 2)
        # Action 1
        transition_tensor[:, 1, 1] = [0.0, 1.0]  # From state 1: Always go to state 2
        transition_tensor[:, 2, 1] = [1.0, 0.0]  # From state 2: Always go to state 1
        # Action 2
        transition_tensor[:, 1, 2] = [1.0, 0.0]  # From state 1: Always stay in state 1
        transition_tensor[:, 2, 2] = [0.0, 1.0]  # From state 2: Always stay in state 2

        observation_matrix = [1.0 0.0; 0.0 1.0]  # Deterministic observations
        reward_states = [(2, 1.0)]

        maze = StochasticMaze(transition_tensor, observation_matrix, reward_states)
        agent = StochasticMazeAgent(1)  # Start in state 1

        # Test action 1 (should move to state 2)
        move!(maze, agent, StochasticMazeAction(1))
        @test agent.state == 2

        # Test action 1 again (should move back to state 1)
        move!(maze, agent, StochasticMazeAction(1))
        @test agent.state == 1

        # Test action 2 (should stay in state 1)
        move!(maze, agent, StochasticMazeAction(2))
        @test agent.state == 1
    end

    # Test reward system
    @testset "Rewards" begin
        transition_tensor = zeros(2, 2, 1)  # Single action
        transition_tensor[:, 1, 1] = [0.0, 1.0]  # From state 1: Always go to state 2
        transition_tensor[:, 2, 1] = [0.0, 1.0]  # From state 2: Stay in state 2

        observation_matrix = [1.0 0.0; 0.0 1.0]
        reward_states = [(2, 1.0)]  # State 2 has reward 1.0

        maze = StochasticMaze(transition_tensor, observation_matrix, reward_states)
        agent = StochasticMazeAgent(1)

        # No reward in state 1
        obs, reward = RxEnvironments.what_to_send(agent, maze, StochasticMazeAction(1))
        @test reward === nothing

        # Move to state 2
        move!(maze, agent, StochasticMazeAction(1))
        obs, reward = RxEnvironments.what_to_send(agent, maze, StochasticMazeAction(1))
        @test reward == 1.0
    end

    # Test RxEnvironments integration
    @testset "RxEnvironments Integration" begin
        transition_tensor = zeros(2, 2, 1)
        transition_tensor[:, 1, 1] = [0.0, 1.0]  # From state 1: Always go to state 2
        transition_tensor[:, 2, 1] = [0.0, 1.0]  # From state 2: Stay in state 2

        observation_matrix = [1.0 0.0; 0.0 1.0]
        reward_states = [(2, 1.0)]

        rx_env, rx_agent = create_stochastic_env(
            transition_tensor,
            observation_matrix,
            reward_states;
            start_state=1
        )

        # Test initial state
        @test rx_env.decorated.agents[1].state == 1

        # Test reset
        reset!(rx_env, 2)
        @test rx_env.decorated.agents[1].state == 2

        # Test action execution
        RxEnvironments.receive!(rx_env.decorated, rx_env.decorated.agents[1], StochasticMazeAction(1))
        @test rx_env.decorated.agents[1].state == 2  # Should stay in state 2
    end
end