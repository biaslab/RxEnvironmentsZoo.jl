@testitem "Maze Environment" begin
    using RxEnvironmentsZoo
    using RxEnvironments
    using Distributions
    import RxEnvironments: keep

    @testset "Basic Environment Creation and Movement" begin
        # Create a simple 3x3 maze with walls
        # Binary encoding: North = 1, West = 2, South = 4, East = 8
        structure = UInt8[6 4 12;
            2 0 8;
            3 1 9]
        reward_pos = (((2, 1), 10),)

        # Create environment and agent
        rx_env, rx_agent = create_environment(Maze, structure, reward_pos)
        env = rx_env.decorated

        # Test initial position
        @test env.agents[1].pos == (1, 1)

        # Test movement in valid direction (East)
        send!(rx_env, rx_agent, MazeAgentAction(East()))
        @test env.agents[1].pos == (2, 1)


        send!(rx_env, rx_agent, MazeAgentAction(North()))
        @test env.agents[1].pos == (2, 2)

        send!(rx_env, rx_agent, MazeAgentAction(West()))
        @test env.agents[1].pos == (1, 2)

        send!(rx_env, rx_agent, MazeAgentAction(West()))
        @test env.agents[1].pos == (1, 2)
    end

    @testset "Observation Sampling" begin
        structure = UInt8[
            9 1 1;
            8 0 0;
            12 4 4
        ]
        reward_pos = (((2, 1), 10),)
        rx_env, rx_agent = create_environment(Maze, structure, reward_pos)
        env = rx_env.decorated

        # Test observation sampling
        observation = sample_observation(env, (1, 1))
        @test length(observation) == 9  # 3x3 maze = 9 possible positions
        @test sum(observation) ≈ 1.0  # Probabilities should sum to 1
        @test count(x -> x > 0, observation) ≥ 1  # At least one non-zero probability
    end

    @testset "Reward Mechanics" begin
        structure = UInt8[6 4 12;
            2 0 8;
            3 1 9]
        reward_pos = (((2, 1), 10),)
        rx_env, rx_agent = create_environment(Maze, structure, reward_pos)
        env = rx_env.decorated

        obs = keep(Any)
        subscribe_to_observations!(rx_agent, obs)
        # Move to reward position
        send!(rx_env, rx_agent, MazeAgentAction(East()))

        observation, rewards = RxEnvironments.data(last(obs))
        @test rewards == 10

        # Move away from reward position
        send!(rx_env, rx_agent, MazeAgentAction(North()))
        observation, rewards = RxEnvironments.data(last(obs))
        @test rewards === nothing
    end

    @testset "Direction Actions" begin
        structure = UInt8[6 4 12;
            2 0 8;
            3 1 9]
        reward_pos = (((2, 1), 10),)
        rx_env, rx_agent = create_environment(Maze, structure, reward_pos)
        env = rx_env.decorated

        # Test all directions
        initial_pos = env.agents[1].pos

        # Test Stay
        send!(rx_env, rx_agent, MazeAgentAction(South()))
        @test env.agents[1].pos == initial_pos

        # Test East (valid move)
        send!(rx_env, rx_agent, MazeAgentAction(East()))
        @test env.agents[1].pos == (2, 1)

        # Test West (valid move back)
        send!(rx_env, rx_agent, MazeAgentAction(West()))
        @test env.agents[1].pos == (1, 1)

        # Test South (blocked by wall)
        send!(rx_env, rx_agent, MazeAgentAction(North()))
        @test env.agents[1].pos == (1, 2)
    end

    @testset "Reset Functionality" begin
        structure = UInt8[
            9 1 1;
            8 0 0;
            12 4 4
        ]
        reward_pos = (((2, 1), 10),)
        rx_env, rx_agent = create_environment(Maze, structure, reward_pos)

        # Move agent around
        send!(rx_env, rx_agent, MazeAgentAction(East()))
        send!(rx_env, rx_agent, MazeAgentAction(North()))

        # Reset to starting position
        reset!(rx_env)
        @test rx_env.decorated.agents[1].pos == (1, 1)

        # Reset to custom position
        reset!(rx_env, (2, 2))
        @test rx_env.decorated.agents[1].pos == (2, 2)
    end
end