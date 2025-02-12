using RxEnvironments
using GLMakie
using Distributions # For Categorical
import Distributions: Categorical

export StochasticMaze, StochasticMazeAgent, StochasticMazeAction, create_stochastic_env
export move!, sample_observation, send_observation_and_reward
export add_agent!, plot_maze_state, reset_agent!, reset!

"""
    StochasticMazeAgent

An agent that can move in a stochastic maze environment.

# Fields
- `state::Int`: Current state index
"""
mutable struct StochasticMazeAgent
    state::Int
end

"""
    StochasticMaze

Represents a stochastic maze environment where transitions are governed by probabilities.

# Fields
- `transition_tensor::Array{Float64,3}`: Transition probabilities (out × in × action)
- `observation_matrix::Matrix{Float64}`: Observation probabilities for each state
- `reward_states::Vector{Tuple{Int,Float64}}`: States and their reward values
- `agents::Vector{StochasticMazeAgent}`: List of agents in the maze
"""
struct StochasticMaze
    transition_tensor::Array{Float64,3}
    observation_matrix::Matrix{Float64}
    reward_states::Vector{Tuple{Int,Float64}}
    agents::Vector{StochasticMazeAgent}
end

"""
    StochasticMaze(transition_tensor, observation_matrix, reward_states)

Construct a StochasticMaze with specified transition probabilities, observation matrix and reward states.
"""
function StochasticMaze(transition_tensor::Array{Float64,3}, observation_matrix::Matrix{Float64}, reward_states::Vector{Tuple{Int,Float64}})
    # Validate transition tensor - each slice should be a valid probability distribution
    for a in 1:size(transition_tensor, 3)
        for s in 1:size(transition_tensor, 2)  # Changed from 1 to 2 for input state dimension
            probs = transition_tensor[:, s, a]  # Changed from [s, :, a] to [:, s, a]
            if !isapprox(sum(probs), 1.0, atol=1e-10) || any(x -> x < 0, probs)
                throw(ArgumentError("Invalid transition probabilities for state $s, action $a"))
            end
        end
    end

    # Validate observation matrix - each column should be a valid probability distribution
    for s in 1:size(observation_matrix, 2)
        probs = observation_matrix[:, s]
        if !isapprox(sum(probs), 1.0, atol=1e-10) || any(x -> x < 0, probs)
            throw(ArgumentError("Invalid observation probabilities for state $s"))
        end
    end

    StochasticMaze(transition_tensor, observation_matrix, reward_states, StochasticMazeAgent[])
end

"""
    StochasticMazeAction

Represents an action in the stochastic maze.

# Fields
- `index::Int`: Action index
"""
struct StochasticMazeAction
    index::Int
end

"""
Move agent to new state based on action by sampling from transition probabilities.
"""
function move!(env::StochasticMaze, agent::StochasticMazeAgent, action::StochasticMazeAction)
    current_state = agent.state
    probs = env.transition_tensor[:, current_state, action.index]
    next_state = rand(Categorical(probs))
    agent.state = next_state
    return next_state
end

# Environment update does nothing
RxEnvironments.update!(::StochasticMaze, dt) = nothing

# Receive action and move agent
RxEnvironments.receive!(maze::StochasticMaze, agent::StochasticMazeAgent, action::StochasticMazeAction) = move!(maze, agent, action)

"""
Sample an observation from the maze's observation matrix at given state.
"""
function sample_observation(env::StochasticMaze, agent::StochasticMazeAgent)
    probs = env.observation_matrix[:, agent.state]
    return rand(Categorical(probs))
end

"""
Get observation and rewards for current state.
"""
function RxEnvironments.what_to_send(agent::StochasticMazeAgent, maze::StochasticMaze, action::StochasticMazeAction)
    # Find if current state has a reward
    reward = nothing
    for (state, value) in maze.reward_states
        if agent.state == state
            reward = value
            break
        end
    end

    observation = sample_observation(maze, agent)
    return (observation, reward)
end

"""
Add agent to environment.
"""
function add_agent!(env::StochasticMaze, initial_state::Int)
    if initial_state < 1 || initial_state > size(env.transition_tensor, 2)
        error("Initial state must be between 1 and $(size(env.transition_tensor, 2))")
    end
    agent = StochasticMazeAgent(initial_state)
    push!(env.agents, agent)
    return agent
end

"""
Reset agent to starting state.
"""
function reset_agent!(agent::StochasticMazeAgent, initial_state::Int)
    agent.state = initial_state
end

"""
Create environment and agent entities.
"""
function create_stochastic_env(transition_tensor::Array{Float64,3}, observation_matrix::Matrix{Float64}, reward_states::Vector{Tuple{Int,Float64}}; start_state::Int=1)
    maze = StochasticMaze(transition_tensor, observation_matrix, reward_states)
    agent = StochasticMazeAgent(start_state)
    rx_env = create_entity(maze; is_active=true)
    rx_agent = add!(rx_env, agent)
    return rx_env, rx_agent
end

"""
Add agent to environment.
"""
function RxEnvironments.add_to_state!(env::StochasticMaze, agent::StochasticMazeAgent)
    push!(env.agents, agent)
end

# Send observation and reward to the agent
function send_observation_and_reward(env::StochasticMaze, agent::StochasticMazeAgent)
    observation = sample_observation(env, agent)
    reward = get_reward(env, agent)
    return observation, reward
end

# Get reward based on current state
function get_reward(env::StochasticMaze, agent::StochasticMazeAgent)
    return agent.state in env.reward_states ? 1.0 : 0.0
end

# Helper function for self-loop visualization
function draw_self_loop!(ax, pos, radius)
    # Draw a circular arrow that loops back to the same position
    θ = range(0, 3π / 2, length=50)  # Leave a gap for arrow head
    center = (pos[1] - 0.1, pos[2] + 0.1)
    circle = [(center[1] + radius * cos(t), center[2] + radius * sin(t)) for t in θ]
    lines!(ax, first.(circle), last.(circle), color=(:black, 0.5))

    # Add arrow head
    arrow_pos = (center[1] + radius * cos(3π / 2), center[2] + radius * sin(3π / 2))
    arrow_dir = (-radius * sin(3π / 2), radius * cos(3π / 2))
    arrows!(ax, [arrow_pos[1]], [arrow_pos[2]],
        [arrow_dir[1] * 0.2], [arrow_dir[2] * 0.2],
        color=(:black, 0.5), arrowsize=10)
end

"""
Plot the stochastic maze state.
Note: This is a simple visualization showing agent's current state.
For more complex visualizations, users should implement their own plotting function.
"""
function RxEnvironments.plot_state(ax, env::StochasticMaze)
    # Use graph layout for states
    n_states = size(env.transition_tensor, 2)
    n_actions = size(env.transition_tensor, 3)

    # Create circular layout for states
    positions = [Point2f(cos(θ), sin(θ)) for θ in range(0, 2π, length=n_states + 1)[1:end-1]]

    # Draw transition probabilities as lines
    for s in 1:n_states, sp in 1:n_states
        max_prob = maximum(env.transition_tensor[sp, s, a] for a in 1:n_actions)
        if max_prob > 0.1  # Only show significant transitions
            if s == sp  # Self-loop case
                draw_self_loop!(ax, positions[s], 0.2)
            else
                lines!(ax, [positions[s], positions[sp]], color=(:black, max_prob), linewidth=2max_prob)
            end
        end
    end

    # Draw states as circles with reward/observation info
    for (i, pos) in enumerate(positions)
        # Base circle
        color = :lightblue
        for (s, r) in env.reward_states
            if s == i
                color = r > 0 ? :green : :red
                break
            end
        end
        scatter!(ax, [pos], color=color, markersize=50)

        # Observation distribution (simplified)
        common_obs = argmax(env.observation_matrix[:, i])
        text!(ax, "O$common_obs", position=pos, align=(:center, :center))
    end

    # Draw agents
    for agent in env.agents
        pos = positions[agent.state]
        scatter!(ax, [pos], color=:gold, markersize=25, marker='★')
    end

    hidedecorations!(ax)
    hidespines!(ax)
    ax.aspect = DataAspect()
end

# Visualization
function plot_maze_state(env::StochasticMaze)
    n_states = size(env.transition_tensor, 2)

    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())

    # Plot states as circles
    for s in 1:n_states
        x = cos(2π * (s - 1) / n_states)
        y = sin(2π * (s - 1) / n_states)

        # Highlight reward states
        if s in env.reward_states
            scatter!(ax, [x], [y], color=:green, markersize=20)
        else
            scatter!(ax, [x], [y], color=:blue, markersize=15)
        end

        # Add state labels
        text!(ax, x, y, text="$s", align=(:center, :center))
    end

    # Plot agents
    for agent in env.agents
        x = cos(2π * (agent.state - 1) / n_states)
        y = sin(2π * (agent.state - 1) / n_states)
        scatter!(ax, [x], [y], color=:red, markersize=10)
    end

    hidedecorations!(ax)
    hidespines!(ax)

    return fig
end

"""
Reset agent to starting position.
"""
function reset!(env::RxEnvironments.RxEntity{StochasticMaze}, start_pos::Int=1)
    reset_agent!(env.decorated.agents[1], start_pos)
end

function create_environment(::Type{StochasticMaze}, transition_tensor::Array{Float64,3}, observation_matrix::Matrix{Float64}, reward_states::Vector{Tuple{Int,Float64}}; start_state::Int=1)
    env = StochasticMaze(transition_tensor, observation_matrix, reward_states)
    rxe = RxEnvironment(env)
    return rxe
end

