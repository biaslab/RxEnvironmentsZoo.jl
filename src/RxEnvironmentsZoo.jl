module RxEnvironmentsZoo

export create_environment
# Write your package code here.
include("environments/pendulum/env.jl")
include("environments/2d_drone/env.jl")
include("environments/stochastic_maze/env.jl")
include("environments/maze/env.jl")
include("environments/mountain_car/env.jl")
include("environments/windy_gridworld/env.jl")

end
