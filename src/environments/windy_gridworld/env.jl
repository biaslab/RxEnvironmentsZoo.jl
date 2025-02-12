using RxEnvironments

export WindyGridWorld, WindyGridWorldAgent, reset_env!, plot_environment

struct WindyGridWorld{N}
    wind::NTuple{N,Int}
    agents::Vector
    goal::Tuple{Int,Int}
end

mutable struct WindyGridWorldAgent
    position::Tuple{Int,Int}
end

RxEnvironments.update!(env::WindyGridWorld, dt) = nothing # The environment has no "internal" updating process over time

function RxEnvironments.receive!(env::WindyGridWorld{N}, agent::WindyGridWorldAgent, action::Tuple{Int,Int}) where {N}
    if action[1] != 0
        @assert action[2] == 0 "Only one of the two actions can be non-zero"
    elseif action[2] != 0
        @assert action[1] == 0 "Only one of the two actions can be non-zero"
    end
    # First check if the new x position would be valid
    new_x = agent.position[1] + action[1]
    if 0 < new_x <= N
        # Only apply wind if x position is valid
        new_position = (new_x, agent.position[2] + action[2] + env.wind[new_x])
        if all(elem -> 0 < elem < N, new_position)
            agent.position = new_position
        end
    end
end

function RxEnvironments.what_to_send(env::WindyGridWorld, agent::WindyGridWorldAgent)
    return agent.position
end

function RxEnvironments.what_to_send(agent::WindyGridWorldAgent, env::WindyGridWorld)
    return agent.position
end

function RxEnvironments.add_to_state!(env::WindyGridWorld, agent::WindyGridWorldAgent)
    push!(env.agents, agent)
end

function reset_env!(environment::RxEnvironments.RxEntity{<:WindyGridWorld,T,S,A}) where {T,S,A}
    env = environment.decorated
    for agent in env.agents
        agent.position = (1, 1)
    end
    for subscriber in RxEnvironments.subscribers(environment)
        send!(subscriber, environment, (1, 1))
    end
end

function plot_environment(environment::RxEnvironments.RxEntity{<:WindyGridWorld,T,S,A}) where {T,S,A}
    env = environment.decorated
    p1 = scatter([env.goal[1]], [env.goal[2]], color=:blue, label="Goal", xlims=(0, 6), ylims=(0, 6))
    for agent in env.agents
        p1 = scatter!([agent.position[1]], [agent.position[2]], color=:red, label="Agent")
    end
    return p1
end