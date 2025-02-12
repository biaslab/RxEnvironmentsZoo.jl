@testitem "Windy Gridworld Environment" begin
    using RxEnvironmentsZoo
    using RxEnvironments

    @testset "Basic Environment Creation" begin
        # Create a simple windy gridworld with 5 columns
        wind = (0, 0, 1, 2, 1)
        goal = (4, 3)
        env = WindyGridWorld(wind, [], goal)

        # Test environment properties
        @test length(env.wind) == 5
        @test env.goal == goal
        @test isempty(env.agents)
    end

    @testset "Agent Movement and Wind Effects" begin
        wind = (0, 0, 1, 2, 1)
        goal = (4, 3)
        env = WindyGridWorld(wind, [], goal)
        agent = WindyGridWorldAgent((1, 1))

        # Add agent to environment
        push!(env.agents, agent)

        # Test horizontal movement (with no wind effect)
        receive!(env, agent, (1, 0))  # Move right
        @test agent.position == (2, 1)

        # Test vertical movement with wind
        receive!(env, agent, (0, 1))  # Move up in column 2
        @test agent.position == (2, 2)  # No wind in column 2

        # Move to windy column and test wind effect
        receive!(env, agent, (1, 0))  # Move to column 3
        @test agent.position[1] == 3  # Horizontal position
        @test agent.position[2] > 2   # Should be pushed up by wind
    end

    @testset "Boundary Conditions" begin
        wind = (0, 0, 1, 2, 1)
        goal = (4, 3)
        env = WindyGridWorld(wind, [], goal)
        agent = WindyGridWorldAgent((1, 1))
        push!(env.agents, agent)

        # Test movement at boundaries
        # Try to move out of bounds
        receive!(env, agent, (-1, 0))  # Try to move left out of bounds
        @test agent.position == (1, 1)  # Should stay in place

        # Move to top and try to go higher
        for _ in 1:5
            receive!(env, agent, (0, 1))
        end
        last_position = agent.position
        receive!(env, agent, (0, 1))  # Try to move beyond top
        @test agent.position == last_position  # Should stay in place
    end

    @testset "Action Validation" begin
        wind = (0, 0, 1, 2, 1)
        goal = (4, 3)
        env = WindyGridWorld(wind, [], goal)
        agent = WindyGridWorldAgent((1, 1))
        push!(env.agents, agent)

        # Test diagonal movement (should not be allowed)
        @test_throws AssertionError receive!(env, agent, (1, 1))
    end

    @testset "Environment Reset" begin
        wind = (0, 0, 1, 2, 1)
        goal = (4, 3)
        env = WindyGridWorld(wind, [], goal)
        agent = WindyGridWorldAgent((1, 1))
        rx_env = RxEnvironment(env)
        add!(rx_env, agent)

        # Move agent
        receive!(env, agent, (1, 0))
        receive!(env, agent, (0, 1))
        @test agent.position != (1, 1)

        # Reset environment
        reset_env!(rx_env)
        @test agent.position == (1, 1)
    end

    @testset "Observation Space" begin
        wind = (0, 0, 1, 2, 1)
        goal = (4, 3)
        env = WindyGridWorld(wind, [], goal)
        agent = WindyGridWorldAgent((1, 1))

        # Test observation format
        obs = RxEnvironments.what_to_send(env, agent)
        @test obs isa Tuple{Int,Int}
        @test obs == agent.position

        # Test agent's observation of environment
        env_obs = RxEnvironments.what_to_send(agent, env)
        @test env_obs == agent.position
    end
end