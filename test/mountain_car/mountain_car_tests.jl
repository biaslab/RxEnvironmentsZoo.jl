@testitem "Mountain Car Environment" begin
    using RxEnvironmentsZoo
    using RxEnvironments
    import RxEnvironmentsZoo: landscape

    @testset "Basic Environment Creation" begin
        import RxEnvironmentsZoo: position, velocity, throttle

        # Create mountain car agent with standard parameters
        initial_position = 0.0
        engine_power = 1.0
        friction_coefficient = 0.1
        mass = 1.0
        target = 1.0

        agent = MountainCarAgent(initial_position, engine_power, friction_coefficient, mass, target)
        env = MountainCarEnvironment(RxEnvironmentsZoo.landscape)

        # Test initial state
        @test position(agent) ≈ 0.0
        @test velocity(agent) ≈ 0.0
        @test throttle(agent) ≈ 0.0
    end

    @testset "State Updates and Dynamics" begin
        agent = MountainCarAgent(0.0, 1.0, 0.1, 1.0, 1.0)
        env = create_environment(MountainCarEnvironment)
        rxagent = add!(env, agent)
        send!(env, rxagent, Throttle(1.0))
        # Test state update with no throttle
        sleep(1)
        # Car should move due to throttle
        @test velocity(agent) > 0.0

        # Apply throttle and check motion
        send!(env, rxagent, Throttle(-1.0))  # Full throttle backward
        sleep(1)
        # Velocity should decrease with negative throttle
        @test velocity(agent) < 0.0
    end

    @testset "Action Clamping" begin
        agent = MountainCarAgent(0.0, 1.0, 0.1, 1.0, 1.0)
        env = MountainCarEnvironment(landscape)

        # Test throttle clamping (should be between -1.0 and 1.0)
        throttle_high = Throttle(2.0)
        @test throttle(throttle_high) ≈ 1.0

        throttle_low = Throttle(-2.0)
        @test throttle(throttle_low) ≈ -1.0
    end



    @testset "Trajectory Recomputation" begin
        import RxEnvironmentsZoo: trajectory, recompute
        agent = MountainCarAgent(0.0, 1.0, 0.1, 1.0, 1.0)
        env = create_environment(MountainCarEnvironment)
        rxagent = add!(env, agent)

        # Initial update to create trajectory
        sleep(0.1)
        initial_trajectory = trajectory(agent)

        # Apply action and check trajectory recomputation
        send!(env, rxagent, Throttle(1.0))
        sleep(0.1)
        @test recompute(trajectory(agent)) == false  # Should be false after computation
        @test trajectory(agent) !== initial_trajectory  # Should be a new trajectory
    end

    @testset "Physical Constraints" begin
        agent = MountainCarAgent(0.0, 1.0, 0.1, 1.0, 1.0)
        env = create_environment(MountainCarEnvironment)
        rxagent = add!(env, agent)

        # Test friction effect
        send!(env, rxagent, Throttle(1.0))  # full
        sleep(0.1)
        v1 = abs(velocity(agent))
        send!(env, rxagent, Throttle(0.0))  # No throttle
        sleep(0.1)
        v2 = abs(velocity(agent))
        @test v2 < v1  # Velocity should decrease due to friction

        # Test gravity effect at different positions
        agent_uphill = MountainCarAgent(0.5, 1.0, 0.0, 1.0, 1.0)  # No friction for this test
        env_uphill = create_environment(MountainCarEnvironment)
        rxagent_uphill = add!(env_uphill, agent_uphill)

        sleep(0.1)
        @test velocity(agent_uphill) < 0.0  # Should roll backwards due to gravity
    end
end