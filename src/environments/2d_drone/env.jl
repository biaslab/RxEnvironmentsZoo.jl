using Distributions
import Distributions: Categorical
using DifferentialEquations
using RxEnvironments
using GLMakie

export add_drone!, TwoDDroneEnvironment

mutable struct DroneEngine{T<:Real}
    max_thrust::T
    min_thrust::T
    current_thrust::T
    desired_thrust::T
    side::Symbol
    recompute::Bool
end

max_thrust(engine::DroneEngine) = engine.max_thrust
min_thrust(engine::DroneEngine) = engine.min_thrust
current_thrust(engine::DroneEngine) = engine.current_thrust
recompute(engine::DroneEngine) = engine.recompute
mutable struct DroneTrajectory{T<:Real}
    recompute::Bool
    time_left::T
    trajectory::Any
    T::T
end


recompute(trajectory::DroneTrajectory) = trajectory.recompute
time_left(trajectory::DroneTrajectory) = trajectory.time_left
current_time(trajectory::DroneTrajectory) = total_time(trajectory) - time_left(trajectory)

total_time(trajectory::DroneTrajectory) = trajectory.T
Base.getindex(trajectory::DroneTrajectory, index) = trajectory.trajectory(index)
set_recompute!(trajectory::DroneTrajectory, recompute) = trajectory.recompute = recompute
set_time_left!(trajectory::DroneTrajectory, time_left) = trajectory.time_left = time_left
reduce_time_left!(trajectory::DroneTrajectory, elapsed_time) =
    set_time_left!(trajectory, time_left(trajectory) - elapsed_time)

mutable struct DroneState{T<:Real}
    x::T
    y::T
    vx::T
    vy::T
    θ::T
    ω::T
    trajectory::DroneTrajectory{T}
end

position(state::DroneState) = [state.x, state.y]
velocity(state::DroneState) = [state.vx, state.vy]
orientation(state::DroneState) = state.θ
angular_velocity(state::DroneState) = state.ω
Base.vec(state::DroneState) = [state.x, state.y, state.vx, state.vy, state.θ, state.ω]
trajectory(state::DroneState) = state.trajectory

set_x!(state::DroneState, x) = state.x = x
set_y!(state::DroneState, y) = state.y = y
set_vx!(state::DroneState, vx) = state.vx = vx
set_vy!(state::DroneState, vy) = state.vy = vy
set_θ!(state::DroneState, θ) = state.θ = θ
set_ω!(state::DroneState, ω) = state.ω = ω
set_state!(state::DroneState, nstate) = begin
    x, y, vx, vy, θ, ω = nstate
    set_x!(state, x)
    set_y!(state, y)
    set_vx!(state, vx)
    set_vy!(state, vy)
    set_θ!(state, θ)
    set_ω!(state, ω)
end
set_trajectory!(state::DroneState, trajectory) = state.trajectory = trajectory

struct DroneBody{T<:Real}
    mass::T
    inertia::T
    radius::T
    state::DroneState{T}
    engines::Vector{DroneEngine{T}}
end

state(body::DroneBody) = body.state
mass(body::DroneBody) = body.mass
inertia(body::DroneBody) = body.inertia
radius(body::DroneBody) = body.radius
engines(body::DroneBody) = body.engines
trajectory(body::DroneBody) = trajectory(state(body))
get_properties(body::DroneBody) = (mass(body), inertia(body), radius(body))
set_position!(state::DroneState, position::Vector) = state.position = position
set_trajectory!(body::DroneBody, trajectory) = set_trajectory!(state(body), trajectory)

struct TwoDDroneEnvironment{T<:Real}
    gravitation::T
    agents::Vector{DroneBody{T}}
end

get_gravity(env::TwoDDroneEnvironment) = env.gravitation
TwoDDroneEnvironment() = TwoDDroneEnvironment(9.81, DroneBody{Float64}[])

struct TwoDDroneAgent
    left_engine_hist::Vector{Float64}
    right_engine_hist::Vector{Float64}
end

TwoDDroneAgent() = TwoDDroneAgent(Float64[], Float64[])

function RxEnvironments.add_to_state!(environment::TwoDDroneEnvironment, agent::DroneBody)
    push!(environment.agents, agent)
end
# Compute forces and torques acting on the drone
function compute_forces_and_torques(m, g, Fl, Fr, θ, r)
    Fg = m * g
    Fy = (Fl + Fr) * cos(θ) - Fg
    Fx = (Fl + Fr) * sin(θ)
    𝜏 = (Fl - Fr) * r
    return Fx, Fy, 𝜏
end

# Compute linear accelerations
function compute_accelerations(Fx, Fy, vx, vy, m)
    ax = (Fx - vx) / m
    ay = (Fy - vy) / m
    return ax, ay
end

# Compute angular acceleration
function compute_angular_acceleration(𝜏, I)
    return 𝜏 / I
end

# DifferentialEquations.jl function describing the environment dynamics.
function __two_d_drone_dynamics(du, u, s, t)
    drone, environment = s

    # extract properties
    m, I, r = get_properties(drone)
    g = get_gravity(environment)
    Fl, Fr = current_thrust.(engines(drone))
    x, y, vx, vy, θ, ω = u

    # compute dynamics
    Fx, Fy, 𝜏 = compute_forces_and_torques(m, g, Fl, Fr, θ, r)
    ax, ay = compute_accelerations(Fx, Fy, vx, vy, m)
    α = compute_angular_acceleration(𝜏, I)

    # update state derivatives
    du[1] = vx
    du[2] = vy
    du[3] = ax
    du[4] = ay
    du[5] = ω
    du[6] = α
end

# Function that computes a trajectory for a mountain car for 5 seconds ahead and saves this result in the state of the corresponding car.
function __compute_drone_dynamics(
    agent::DroneBody,
    environment::TwoDDroneEnvironment,
)
    T = 5.0
    initial_state = vec(state(agent))
    tspan = (0.0, T)
    prob = ODEProblem(__two_d_drone_dynamics, initial_state, tspan, (agent, environment))
    sol = solve(prob, Tsit5())
    set_trajectory!(agent, DroneTrajectory(false, T, sol, T))
end

function RxEnvironments.update!(environment::TwoDDroneEnvironment, dt::Real)
    # Update all actors in the environment
    for agent in environment.agents
        if recompute(trajectory(agent)) || time_left(trajectory(agent)) < dt || any(recompute.(engines(agent)))
            __compute_drone_dynamics(agent, environment)
        end
        reduce_time_left!(trajectory(agent), dt)
        new_state = trajectory(agent)[current_time(trajectory(agent))]
        set_state!(state(agent), new_state)
    end
end

RxEnvironments.what_to_send(body::DroneBody, environment::TwoDDroneEnvironment) = vec(state(body))
RxEnvironments.emits(body::DroneBody, environment::TwoDDroneEnvironment, any) = false

RxEnvironments.what_to_send(engine::DroneEngine, body::DroneBody) = vec(state(body))
RxEnvironments.emits(engine::DroneEngine, body::DroneBody, any::Real) = begin
    set_recompute!(trajectory(state(body)), true)
    false
end
RxEnvironments.emits(engine::DroneEngine, body::DroneBody, any) = false
RxEnvironments.emits(engine::DroneEngine, agent::TwoDDroneAgent, any::Real) = true

RxEnvironments.receive!(engine::DroneEngine, agent::TwoDDroneAgent, action::Real) = begin
    if engine.side == :left
        push!(agent.left_engine_hist, engine.current_thrust)
    else
        push!(agent.right_engine_hist, engine.current_thrust)
    end
    actual_thrust = clamp(action, min_thrust(engine), max_thrust(engine))
    engine.current_thrust = actual_thrust
    engine.desired_thrust = action
    engine.recompute = true
end

DroneBody(mass, inertia, radius) = begin
    state = DroneState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DroneTrajectory(false, 0.0, [], 0.0))
    engines = [DroneEngine(15.0, -15.0, 4.905, 4.905, side, true) for side in [:left, :right]]
    DroneBody(mass, inertia, radius, state, engines)
end

RxEnvironments.plot_state(ax, env::TwoDDroneEnvironment) = begin
    xlims!(ax, -2, 2)
    ylims!(ax, -2, 2)

    colors = [:red, :blue, :green, :yellow, :purple, :orange, :brown, :pink, :cyan, :magenta]
    for (i, agent) in enumerate(env.agents)
        x, y, x_a, y_a, θ, ω = vec(state(agent))
        _, _, radius = get_properties(agent)
        dx = radius * cos(θ)
        dy = radius * sin(θ)

        drone_position = [x], [y]
        drone_engines = [x - dx, x + dx], [y + dy, y - dy]
        drone_coordinates = [x - dx, x, x + dx], [y + dy, y, y - dy]

        rotation_matrix = [cos(-θ) -sin(-θ); sin(-θ) cos(-θ)]
        engine_shape = [-1 0 1; 1 -1 1]
        drone_shape = [-2 -2 2 2; -1 1 1 -1]

        engine_shape = rotation_matrix * engine_shape
        drone_shape = rotation_matrix * drone_shape

        color = colors[i]
        # @show drone_position
        # @show drone_engines
        # @show drone_coordinates

        scatter!(ax, drone_position[1], drone_position[2]; color=color, marker=:rect)
        scatter!(ax, drone_engines[1], drone_engines[2]; color=color, marker=:dtriangle)
        lines!(ax, drone_coordinates[1], drone_coordinates[2]; color=color)
    end
end

create_environment(::Type{TwoDDroneEnvironment}; emit_every_ms=20) = begin
    env = TwoDDroneEnvironment()
    rxe = RxEnvironment(env; emit_every_ms=emit_every_ms)
    animate_state(rxe)
    return rxe
end

add_drone!(env) = begin
    body = DroneBody(1.0, 1.0, 0.2)
    rxb = add!(env, body)
    lengine, rengine = engines(body)
    rxl = add!(rxb, lengine)
    rxr = add!(rxb, rengine)
    agent = TwoDDroneAgent()
    rxa = add!(rxl, agent)
    add!(rxa, rxr)
    return rxl, rxr, rxa
end

RxEnvironments.plot_state(ax, agent::TwoDDroneAgent) = begin
    x = min(length(agent.left_engine_hist), 30)
    left = agent.left_engine_hist[end-x+1:end]
    right = agent.right_engine_hist[end-x+1:end]
    lines!(ax, 1:x, left, label="Left")
    lines!(ax, 1:x, right, label="Right")
end