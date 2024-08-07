using DifferentialEquations
using RxEnvironments
using GLMakie

mutable struct ObservablePendulumState{T<:Real}
    theta::T
    angular_velocity::T
    torque::T
end

mutable struct PendulumState{T,D}
    observable_state::ObservablePendulumState{T}
    sol::D
    time_left::T
    time::T
end

theta(state::PendulumState) = state.observable_state.theta
angular_velocity(state::PendulumState) = state.observable_state.angular_velocity

mutable struct Pendulum{T,R,D}
    g::R
    problem::T
    state::PendulumState{R,D}
end

Pendulum(g) = begin
    problem = ODEProblem(__pendulum_dynamics, [0.0, 0.0, 0.0], (0.0, 10.0), g)
    state = PendulumState(ObservablePendulumState(0.0, 0.0, 0.0), solve(problem), 10.0, 0.0)
    Pendulum(g, problem, state)
end


function __pendulum_dynamics(du, u, parameters, t)

    theta, angular_velocity, torque = u
    g, = parameters
    du[1] = angular_velocity
    du[2] = -((3 * g / (2) * cos(theta) + 3.0 * torque)) - 0.5 * angular_velocity
    du[3] = 0.0
end

function __recompute_pendulum_dynamics(pendulum::Pendulum)
    sol = solve(remake(pendulum.problem, u0=[theta(pendulum.state), angular_velocity(pendulum.state), pendulum.state.observable_state.torque]))
    pendulum.state.time_left = 10.0
    pendulum.state.time = 0.0
    pendulum.state.sol = sol
end

struct PendulumAgent

end

RxEnvironments.what_to_send(agent::PendulumAgent, environment::Pendulum) = (cos(theta(environment.state)), sin(theta(environment.state)), angular_velocity(environment.state))

RxEnvironments.receive!(environment::Pendulum, agent::PendulumAgent, action::Real) = begin
    action = clamp(action, -2.0, 2.0)
    environment.state.observable_state.torque = action
    __recompute_pendulum_dynamics(environment)
end

RxEnvironments.update!(environment::Pendulum, elapsed_time::Real) = begin
    environment.state.time_left -= elapsed_time
    environment.state.time += elapsed_time
    if environment.state.time_left <= 0.0
        __recompute_pendulum_dynamics(environment)
    end
    new_state = environment.state.sol(environment.state.time)
    environment.state.observable_state.theta = new_state[1]
    environment.state.observable_state.angular_velocity = new_state[2]
end

RxEnvironments.plot_state(ax, environment::Pendulum) = begin
    xlims!(ax, -1.5, 1.5)
    ylims!(ax, -1.5, 1.5)
    x, y = cos(theta(environment.state)), sin(theta(environment.state))
    lines!(ax, [0, x], [0, y])
end

p = Pendulum(9.81);
rxe = RxEnvironment(p, emit_every_ms=60)
rxa = add!(rxe, PendulumAgent());
animate_state(rxe)
send!(rxe, rxa, -1.0)
send!(rxe, rxa, 0.0)