This environment can be modelled as an MDP: the states 
correspond to the adventurer’s (x, y) position in the maze and
whether he currently possesses the key. Rewards are 0 on every time step
the agent has not reached the end with the key, 1 for successfully
completing the maze, and -100 if the agent touches fire. 
The agent would like to finish the maze as quickly as possible,
so all rewards have a discount rate γ = 0.95.

Used reinforcement learning to help the agent solve the maze as quickly as possible. 
Implemented  Q-learning algorithm. 
The agent will use a epsilon-greedy behaviour policy to ensure adequate exploration of the maze.
The constants γ, ε and step size are given as attributes of the agent. 

Program can be tested by running the module defined in main.py. Program can also be run using python3 main.py. 
An automated graphic will display a final episode after training has been completed. 
A learning curve, showing the length of episodes over time will be generated in learning curve.png.

In the function, display_final_episode(),  the agent will be evaluated using a greedy policy, instead of an ε-greedy policy.

![alt text](https://ibb.co/PWgbzvB)




