import HypergeometricFunctions: _₂F₁
using Distributions
using ForwardDiff
using DifferentialEquations
using LinearAlgebra
using RxEnvironments

export MountainCarEnvironment, MountainCarAgent, Throttle

function landscape(x)
    if x < 0
        h = x^2 + x
    else
        h =
            x * _₂F₁(0.5, 0.5, 1.5, -5 * x^2) +
            x^3 * _₂F₁(1.5, 1.5, 2.5, -5 * x^2) / 3 +
            x^5 / 80
    end
    return 0.05 * h
end

mutable struct MountainCarTrajectory{T<:Real}
    recompute::Bool
    time_left::T
    trajectory::Any
    T::T
end

# Convenient getters and setters
recompute(trajectory::MountainCarTrajectory) = trajectory.recompute
time_left(trajectory::MountainCarTrajectory) = trajectory.time_left
current_time(trajectory::MountainCarTrajectory) =
    total_time(trajectory) - time_left(trajectory)
total_time(trajectory::MountainCarTrajectory) = trajectory.T
Base.getindex(trajectory::MountainCarTrajectory, index) = trajectory.trajectory(index)


set_recompute!(trajectory::MountainCarTrajectory, recompute) =
    trajectory.recompute = recompute
set_time_left!(trajectory::MountainCarTrajectory, time_left) =
    trajectory.time_left = time_left
reduce_time_left!(trajectory::MountainCarTrajectory, elapsed_time) =
    set_time_left!(trajectory, time_left(trajectory) - elapsed_time)

mutable struct MountainCarState{T<:Real}
    position::T
    velocity::T
    throttle::T
    trajectory::MountainCarTrajectory{T}
end

# Convenient getters and setters
position(state::MountainCarState) = state.position
velocity(state::MountainCarState) = state.velocity
throttle(state::MountainCarState) = state.throttle
observable_state(state::MountainCarState) = [position(state), velocity(state)]

set_position!(state::MountainCarState, position::Real) = state.position = position
set_velocity!(state::MountainCarState, velocity::Real) = state.velocity = velocity
set_throttle!(state::MountainCarState, throttle::Real) = state.throttle = throttle
set_trajectory!(state::MountainCarState, trajectory) = state.trajectory = trajectory
trajectory(state::MountainCarState) = state.trajectory

# Convenient constructor that creates an empty trajectory that will immediately be replaced.
MountainCarState(position::Real, velocity::Real, throttle::Real) = MountainCarState(
    position,
    velocity,
    throttle,
    MountainCarTrajectory(true, 0.0, [], 0.0),
)

struct MountainCarAgent{T<:Real}
    state::MountainCarState{T}
    engine_power::T
    friction_coefficient::T
    mass::T
    target::T
end

MountainCarAgent(
    position::Real,
    engine_power::Real,
    friction_coefficient::Real,
    mass::Real,
    target::Real,
) = MountainCarAgent(
    MountainCarState(position, 0.0, 0.0),
    engine_power,
    friction_coefficient,
    mass,
    target,
)

# Convenient getters and setters
state(car::MountainCarAgent) = car.state
position(car::MountainCarAgent) = position(state(car))
velocity(car::MountainCarAgent) = velocity(state(car))
throttle(car::MountainCarAgent) = throttle(state(car))
mass(car::MountainCarAgent) = car.mass
observable_state(car::MountainCarAgent) = observable_state(state(car))

set_position!(car::MountainCarAgent, position::Real) = set_position!(state(car), position)
set_velocity!(car::MountainCarAgent, velocity::Real) = set_velocity!(state(car), velocity)
set_throttle!(car::MountainCarAgent, throttle::Real) = set_throttle!(state(car), throttle)
engine_power(car::MountainCarAgent) = car.engine_power
friction_coefficient(car::MountainCarAgent) = car.friction_coefficient
set_trajectory!(car::MountainCarAgent, trajectory) = set_trajectory!(state(car), trajectory)
trajectory(car::MountainCarAgent) = trajectory(state(car))

struct MountainCarEnvironment
    actors::Vector{MountainCarAgent}
    landscape::Any
end

MountainCarEnvironment(landscape) = MountainCarEnvironment([], landscape)

struct Throttle{T<:Real}
    throttle::T
    Throttle(throttle::T) where {T<:Real} = new{T}(clamp(throttle, -1, 1))
end

function RxEnvironments.add_to_state!(environment::MountainCarEnvironment, agent::MountainCarAgent)
    push!(environment.actors, agent)
end

throttle(action::Throttle) = action.throttle
friction(car::MountainCarAgent, velocity) = velocity * -friction_coefficient(car)
gravitation(car::MountainCarAgent, position, landscape) =
    mass(car) * -9.81 * sin(atan(ForwardDiff.derivative(landscape, position)))

# DifferentialEquations.jl function describing the environment dynamics.
function __mountain_car_dynamics(du, u, s, t)
    agent, env = s
    position, momentum = u
    du[1] = momentum
    du[2] =
        throttle(agent) +
        friction(agent, momentum) +
        gravitation(agent, position, env.landscape)
end

# Function that computes a trajectory for a mountain car for 5 seconds ahead and saves this result in the state of the corresponding car.
function __compute_mountain_car_dynamics(
    agent::MountainCarAgent,
    environment::MountainCarEnvironment,
)
    T = 5.0
    initial_state = [position(agent), velocity(agent)]
    tspan = (0.0, T)
    prob = ODEProblem(__mountain_car_dynamics, initial_state, tspan, (agent, environment))
    sol = solve(prob, Tsit5())
    set_trajectory!(agent, MountainCarTrajectory(false, T, sol, T))
end

# The agent observes a noisy estimate of its actual position and velocity
RxEnvironments.what_to_send(agent::MountainCarAgent, environment::MountainCarEnvironment) =
    return rand(MvNormal(observable_state(agent), I(2)))

function RxEnvironments.receive!(
    environment::MountainCarEnvironment,
    agent::MountainCarAgent,
    action::Throttle,
)
    # We always have to recompute the trajecory of an agent if this agent conducts an action
    set_recompute!(trajectory(agent), true)
    set_throttle!(agent, throttle(action) * engine_power(agent))
end

function RxEnvironments.update!(environment::MountainCarEnvironment, elapsed_time::Real)
    # Update all actors in the environment
    for agent in environment.actors
        # If we have conducted an action or the current trajecory is valid for less than `elapsed_time` seconds, recompute
        if recompute(trajectory(agent)) || time_left(trajectory(agent)) < elapsed_time
            __compute_mountain_car_dynamics(agent, environment)
        end
        # Bookkeeping for the trajecory, position and velocity
        reduce_time_left!(trajectory(agent), elapsed_time)
        new_state = trajectory(agent)[current_time(trajectory(agent))]
        set_position!(agent, new_state[1])
        set_velocity!(agent, new_state[2])
    end
end


function create_environment(::Type{MountainCarEnvironment}; emit_every_ms=20)
    env = MountainCarEnvironment(landscape)
    rxe = RxEnvironment(env; emit_every_ms=emit_every_ms)
    return rxe
end

