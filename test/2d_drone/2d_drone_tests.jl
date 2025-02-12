@testitem "2D Drone Environment" begin

    @testset "Sanity tests" begin
        using RxEnvironmentsZoo
        using RxEnvironments
        import RxEnvironmentsZoo: TwoDDroneEnvironment

        env = TwoDDroneEnvironment()
        rxe = RxEnvironment(env; emit_every_ms=20)
        left, right, agent = add_drone!(rxe)
        send!(left, agent, 10.0)
        send!(right, agent, 10.0)
        sleep(1)
        @test env.agents[1].state.y > 0.0
        terminate!(rxe)
    end

    @testset "compute_forces_and_torques" begin
        using RxEnvironmentsZoo
        import RxEnvironmentsZoo: compute_forces_and_torques

        # Test basic force calculations
        m = 1.0  # mass
        g = 9.81 # gravity
        Fl = 10.0 # left engine thrust
        Fr = 10.0 # right engine thrust
        θ = 0.0  # angle
        r = 0.2  # radius

        Fx, Fy, τ = compute_forces_and_torques(m, g, Fl, Fr, θ, r)

        @test Fx ≈ 0.0 atol = 1e-10 # No horizontal force when θ=0
        @test Fy ≈ 20.0 - m * g # Vertical force is thrust minus weight
        @test τ ≈ 0.0 atol = 1e-10 # No torque when thrusts are equal

        # Test with angle
        θ = π / 4
        Fx, Fy, τ = compute_forces_and_torques(m, g, Fl, Fr, θ, r)
        @test Fx ≈ 20.0 * sin(θ)
        @test Fy ≈ 20.0 * cos(θ) - m * g

        # Test with unequal thrusts
        Fl = 12.0
        Fr = 8.0
        Fx, Fy, τ = compute_forces_and_torques(m, g, Fl, Fr, θ, r)
        @test τ ≈ (Fl - Fr) * r
    end

    @testset "compute_accelerations" begin
        using RxEnvironmentsZoo
        import RxEnvironmentsZoo: compute_accelerations

        # Test basic acceleration calculations
        Fx = 10.0
        Fy = 20.0
        vx = 2.0
        vy = 3.0
        m = 2.0

        ax, ay = compute_accelerations(Fx, Fy, vx, vy, m)

        @test ax ≈ (Fx - vx) / m
        @test ay ≈ (Fy - vy) / m

        # Test with zero velocity
        ax, ay = compute_accelerations(Fx, Fy, 0.0, 0.0, m)
        @test ax ≈ Fx / m
        @test ay ≈ Fy / m

        # Test with zero force
        ax, ay = compute_accelerations(0.0, 0.0, vx, vy, m)
        @test ax ≈ -vx / m
        @test ay ≈ -vy / m
    end

    @testset "compute_angular_acceleration" begin
        using RxEnvironmentsZoo
        import RxEnvironmentsZoo: compute_angular_acceleration

        # Test basic angular acceleration calculations
        τ = 2.0
        I = 1.0
        α = compute_angular_acceleration(τ, I)
        @test α ≈ τ / I

        # Test with different inertia
        I = 2.0
        α = compute_angular_acceleration(τ, I)
        @test α ≈ τ / I

        # Test with zero torque
        α = compute_angular_acceleration(0.0, I)
        @test α ≈ 0.0 atol = 1e-10
    end
end