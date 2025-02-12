using RxEnvironments
using GLMakie
using Distributions # For Categorical

export North, East, South, West, MazeAgentAction, animate_state, create_transition_tensor, Maze

"""
    MazeAgent

An agent that can move in a maze environment.

# Fields
- `pos::Tuple{Int,Int}`: Current position as (x,y) tuple
"""
mutable struct MazeAgent
    pos::Tuple{Int,Int}
end

"""
    Maze

Represents a general maze environment.

# Fields
- `structure::Matrix{UInt8}`: Binary encoding of walls for each cell
- `observation_matrix::Matrix{Float64}`: Observation probabilities for each position
- `reward_pos::NTuple{N,Tuple{Tuple{Int,Int},Int}}`: Positions and values of rewards
- `agents::Vector{MazeAgent}`: List of agents in the maze
"""
struct Maze
    structure::Matrix{UInt8}
    observation_matrix::Matrix{Float64}
    reward_pos::NTuple{N,Tuple{Tuple{Int,Int},Int}} where {N}
    agents::Vector{MazeAgent}
end

"""
    Maze(structure, observation_matrix, rewardpos)

Construct a Maze with specified structure, observation matrix and reward positions.
"""
function Maze(structure::Matrix{UInt8}, observation_matrix::Matrix{Float64}, rewardpos)
    Maze(structure, observation_matrix, rewardpos, MazeAgent[])
end

"""
    Maze(structure, rewardpos)

Construct a Maze with specified structure and reward positions, using default observation matrix.
"""
function Maze(structure::Matrix{UInt8}, rewardpos)
    width, height = size(structure)
    observation_matrix = create_default_observation_matrix(width, height)
    Maze(structure, observation_matrix, rewardpos, MazeAgent[])
end

"""
Get the boundary encoding for a position in the maze.
"""
boundaries(maze::Maze, pos::NTuple{2,Int}) = maze.structure[pos[2], pos[1]]

# Direction types for agent actions
struct North end
struct East end
struct South end
struct West end
struct Stay end

const DIRECTIONS = (North(), East(), South(), West(), Stay())

"""
    MazeAgentAction

Represents a directional action in the maze.

# Fields
- `direction`: One of North(), East(), South(), West(), Stay()
"""
struct MazeAgentAction
    direction::Union{North,East,South,West,Stay}
end

"""
Move agent to new position based on action.
"""
function move!(agent::MazeAgent, maze::Maze, a::Union{MazeAgentAction,Int})
    agent.pos = next_state(agent.pos, maze, a)
end

"""
Convert numeric action (1-5) to directional action and get next state.
"""
function next_state(agent_pos::Tuple{Int,Int}, maze::Maze, action::Int)
    1 <= action <= 5 || throw(ArgumentError("Action must be between 1 and 5"))
    return next_state(agent_pos, maze, MazeAgentAction(DIRECTIONS[action]))
end

"""
Get next state based on agent position, maze boundaries and action.
"""
function next_state(agent_pos::Tuple{Int,Int}, maze::Maze, action::MazeAgentAction)
    c_bounds = boundaries(maze, agent_pos)
    b_bounds = digits(c_bounds, base=2, pad=4)  # Convert to binary wall encoding
    return next_state(agent_pos, b_bounds, action.direction)
end

"""
Move north if no wall, otherwise stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::North)
    if b_bounds[1] == 0
        return (agent_pos[1], agent_pos[2] + 1)
    else
        return agent_pos
    end
end

"""
Move west if no wall, otherwise stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::West)
    if b_bounds[2] == 0
        return (agent_pos[1] - 1, agent_pos[2])
    else
        return agent_pos
    end
end

"""
Move south if no wall, otherwise stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::South)
    if b_bounds[3] == 0
        return (agent_pos[1], agent_pos[2] - 1)
    else
        return agent_pos
    end
end

"""
Move east if no wall, otherwise stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::East)
    if b_bounds[4] == 0
        return (agent_pos[1] + 1, agent_pos[2])
    else
        return agent_pos
    end
end

"""
Stay in place.
"""
function next_state(agent_pos::Tuple{Int,Int}, b_bounds, ::Stay)
    return agent_pos
end

# Environment update does nothing
RxEnvironments.update!(::Maze, dt) = nothing

# Receive action and move agent
RxEnvironments.receive!(maze::Maze, agent::MazeAgent, action::Union{MazeAgentAction,Int}) = move!(agent, maze, action)

"""
Convert linear index to (x,y) coordinates.
"""
function index_to_coord(index::Int, width::Int)::Tuple{Int,Int}
    y = div(index - 1, width) + 1
    x = mod(index - 1, width) + 1
    return (x, y)
end

"""
Convert (x,y) coordinates to linear index.
"""
function coord_to_index(x::Int, y::Int, width::Int)::Int
    return (y - 1) * width + x
end

"""
Sample an observation from the maze's observation matrix at given position.
"""
function sample_observation(maze::Maze, pos::Tuple{Int,Int})
    index = coord_to_index(pos[1], pos[2], size(maze.structure, 2))
    observation_probs = maze.observation_matrix[:, index]
    dist = Categorical(observation_probs)
    sampled_index = rand(dist)
    onehot = zeros(Float64, length(maze.structure))
    onehot[sampled_index] = 1.0
    return onehot
end

"""
Create a default observation matrix with noise increasing with distance from origin.
This is just one possible way to create an observation matrix - users can provide their own.
"""
function create_default_observation_matrix(width::Int, height::Int; noise_scale::Float64=0.1)
    size = width * height
    observation_matrix = zeros(Float64, size, size)
    for i in 1:size
        index = index_to_coord(i, width)
        # Calculate noise based on Manhattan distance from current position
        noise = (abs(width - index[1]) + abs(height - index[2])) * noise_scale
        noise_factor = noise / (size - 1)
        probvec = fill(noise_factor, size)
        probvec[i] = 1 - noise
        # Normalize to ensure probabilities sum to 1
        observation_matrix[:, i] = probvec ./ sum(probvec)
    end
    return observation_matrix
end

"""
Create transition tensor encoding state transitions for each action.
"""
function create_transition_tensor(maze::Maze)
    width, height = size(maze.structure)
    maze_size = width * height
    transition_tensor = zeros(Float64, maze_size, maze_size, 5)

    # Initialize all states to stay in place for the Stay action (index 5)
    transition_tensor[:, :, 5] = diagm(ones(maze_size))

    # Handle directional actions (indices 1-4)
    for i in 1:maze_size
        pos = index_to_coord(i, width)
        for a in 1:4
            new_pos = next_state(pos, maze, a)
            new_pos_index = coord_to_index(new_pos[1], new_pos[2], width)
            transition_tensor[i, new_pos_index, a] = 1
        end
    end
    return transition_tensor
end

"""
Get observation and rewards for current state.
"""
function RxEnvironments.what_to_send(agent::MazeAgent, maze::Maze, action::MazeAgentAction)
    rewards = nothing
    for reward_loc in maze.reward_pos
        if agent.pos == reward_loc[1]
            rewards = reward_loc[2]
            break
        end
    end
    observation = sample_observation(maze, agent.pos)
    return (observation, rewards)
end

"""
Plot cell walls based on boundary encoding.
"""
function plot_cell!(ax, cell, pos, internal=false)
    cell_bounds = digits(cell, base=2, pad=4)
    wall_coords = [
        ((pos[1] - 0.5, pos[2] + 0.5), (pos[1] + 0.5, pos[2] + 0.5)), # North
        ((pos[1] - 0.5, pos[2] - 0.5), (pos[1] - 0.5, pos[2] + 0.5)), # West
        ((pos[1] - 0.5, pos[2] - 0.5), (pos[1] + 0.5, pos[2] - 0.5)), # South
        ((pos[1] + 0.5, pos[2] - 0.5), (pos[1] + 0.5, pos[2] + 0.5))  # East
    ]

    for (i, ((x1, y1), (x2, y2))) in enumerate(wall_coords)
        if cell_bounds[i] == 1
            lines!(ax, [x1, x2], [y1, y2], color="black")
        elseif internal
            lines!(ax, [x1, x2], [y1, y2], color="black", linestyle=:dash)
        end
    end
end

"""
Add agent to environment.
"""
function RxEnvironments.add_to_state!(env::Maze, agent::MazeAgent)
    push!(env.agents, agent)
end

"""
Plot the maze and agents.
"""
function RxEnvironments.plot_state(ax, env::Maze)
    ys, xs = size(env.structure)
    xlims!(ax, -0.5, xs + 0.5)
    ylims!(ax, -0.5, ys + 0.5)

    # Plot maze cells
    for x in 1:xs, y in 1:ys
        plot_cell!(ax, env.structure[y, x], (x - 0.5, (y - 0.5)), false)
    end

    # Plot agents
    for agent in env.agents
        scatter!(ax, [agent.pos[1] - 0.5], [agent.pos[2] - 0.5], color="red")
    end
end

"""
Create environment and agent entities.
"""
function create_environment(::Type{Maze}, structure::Matrix{UInt8}, reward_pos; start_pos::Tuple{Int,Int}=(1, 1))
    maze = Maze(structure, reward_pos)
    agent = MazeAgent(start_pos)
    rx_env = create_entity(maze; is_active=true)
    rx_agent = add!(rx_env, agent)
    return rx_env, rx_agent
end

"""
Reset agent to starting position.
"""
function reset!(env::RxEnvironments.RxEntity{Maze}, start_pos::Tuple{Int,Int}=(1, 1))
    env.decorated.agents[1].pos = start_pos
end