@testitem "Pendulum Environment" begin
    using RxEnvironmentsZoo
    using RxEnvironments
    @testset "Basic Environment Creation" begin
        import RxEnvironmentsZoo: theta, angular_velocity
        # Create pendulum with standard gravity
        g = 9.81
        pendulum = Pendulum(g)

        # Test initial state
        @test theta(pendulum.state) ≈ -pi / 2
        @test angular_velocity(pendulum.state) ≈ 0.0
        @test pendulum.state.observable_state.torque ≈ 0.0
    end

    @testset "State Updates and Dynamics" begin
        pendulum = Pendulum(9.81)
        dt = 1.0

        # Test state update with no torque
        RxEnvironments.update!(pendulum, dt)
        # Under gravity with no torque, pendulum should stay at equilibrium
        @test abs(theta(pendulum.state) + (pi / 2)) < 1e-10
        @test abs(angular_velocity(pendulum.state)) < 1e-10

        # Apply torque and check motion
        agent = PendulumAgent()
        receive!(pendulum, agent, 1.0)
        update!(pendulum, dt)
        # Positive torque should create positive angular velocity
        @test angular_velocity(pendulum.state) > 0.0
    end

    @testset "Action Clamping" begin
        pendulum = Pendulum(9.81)
        agent = PendulumAgent()

        # Test action clamping (should be between -2.0 and 2.0)
        receive!(pendulum, agent, 5.0)  # Should be clamped to 2.0
        @test pendulum.state.observable_state.torque ≈ 2.0

        receive!(pendulum, agent, -5.0)  # Should be clamped to -2.0
        @test pendulum.state.observable_state.torque ≈ -2.0
    end

    @testset "Observation Space" begin
        pendulum = Pendulum(9.81)
        agent = PendulumAgent()

        # Test observation format
        obs = RxEnvironments.what_to_send(agent, pendulum)
        @test length(obs) == 2  # Should return (θ, ω)
    end

    @testset "Trajectory Recomputation" begin
        pendulum = Pendulum(9.81)
        agent = PendulumAgent()

        # Test trajectory recomputation after long time
        initial_sol = pendulum.state.sol
        update!(pendulum, 11.0)  # Beyond the 10.0 time horizon
        @test pendulum.state.sol !== initial_sol  # Should have recomputed trajectory
    end
end