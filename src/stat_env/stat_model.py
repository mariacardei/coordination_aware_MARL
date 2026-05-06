"""Core STAT simulation.

STAT is a domain-general spatial task-allocation testbed. Public APIs and
documentation use:

- agents: mobile decision makers
- tasks: spatially distributed work items
- execute task: the terminal service action for a selected task

The active RL policy used by the release artifact is policy 6, where external
DQN/FDQN/PyMARL code provides masked task-allocation actions.
"""
import mesa
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
import numpy as np
import random
import math
import itertools

DEBUG = False


def compute_completed(model):
    """Return the number of completed tasks.

    The underlying task flag is named `completed`.
    """
    return sum(task.completed for task in model.sorted_tasks)

def get_distance(pos1, pos2):
    """Return Euclidean distance between two positions."""
    x_1, y_1 = pos1
    x_2, y_2 = pos2
    return math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)


def calc_move(x_1, y_1, x_2, y_2, distance):
    """Move from one point toward another without overshooting."""
    delta_y = y_2 - y_1
    delta_x = x_2 - x_1
    if delta_y != 0:
        ratio = delta_x / delta_y
        delta_y = float(math.sqrt(distance**2 / (ratio**2 + 1))) * (-1 if delta_y < 0 else 1)
        delta_x = float(abs(ratio * delta_y)) * (-1 if delta_x < 0 else 1)
    else:
        delta_y = 0.0
        delta_x = float(distance) * (-1 if delta_x < 0 else 1)

    if abs(delta_x) > abs(x_2 - x_1) or abs(delta_y) > abs(y_2 - y_1):
        return (float(x_2), float(y_2)), delta_x, delta_y
    else:
        return (float(x_1 + delta_x), float(y_1 + delta_y)), delta_x, delta_y


class STATAgent(mesa.Agent):
    """Mobile STAT agent.

    Domain-general role: an agent that selects, moves to, and completes tasks.
    """

    IDLE = 0
    MOVING = 1
    EXECUTING = 2
    
    TIME_TO_EXECUTE = 3
    
    def __init__(self, unique_id, model, policy, speed=1.0):
        super().__init__(unique_id, model)
        self.speed = speed
        self.target = None
        self.state = STATAgent.IDLE
        self.state_time_counter = self.model.global_time_to_execute_RL
        self.policy = policy
        self.num_completed = 0

        if policy == 5 or policy == 6:
            self.indiv_reward = 0
            
        # Policies 0-4 are scripted task-selection baselines.
        # 5 = tabular Q-learning, 6 = externally controlled RL policy.

    def get_valid_actions(self):
        """
        Returns a list of valid paper actions for the agent.

        - 0: idle
        - 1: move toward selected task
        - 2: execute selected task
        - 3+i: select task i

        The logic is as follows:
        - If the agent already has a selected task:
            - If it is at the task: force execute task ([2]).
            - Otherwise, only allow move ([1]).
        - If no target is currently assigned:
            - Allow selecting from incomplete/unassigned tasks using actions 3+.
            - If no valid task is available, allow idle ([0]).
        """
        valid_actions = []
        
        if self.target:
            if self.at_target():
                if DEBUG:
                    print(f"[DEBUG] Agent {self.unique_id} at task {self.target.unique_id}. Valid action: execute task (2)")
                return [2]
            else:
                if DEBUG:
                    print(f"[DEBUG] Agent {self.unique_id} not at task {self.target.unique_id}. Valid action: move (1)")
                return [1]
        
        else:
            all_tasks_unavailable = all(task.completed == 1 or task.selected for task in self.model.sorted_tasks)
            if all_tasks_unavailable:
                if DEBUG:
                    print(f"[DEBUG] Agent {self.unique_id}: all tasks unavailable, valid action: idle (0)")
                return [0]
            
            for idx, task in enumerate(self.model.sorted_tasks):
                if task.completed == 0 and not task.selected:
                    valid_actions.append(3 + idx)
            if DEBUG:
                print(f"[DEBUG] Agent {self.unique_id}: valid select-task actions: {valid_actions}")
            if not valid_actions:
                valid_actions.append(0)
        if DEBUG:
            print(f"[DEBUG] Agent {self.unique_id}: no selected task; valid actions: {valid_actions}")
        return valid_actions

    def perform_action(self, action):
        """
        Perform the given action and return the reward and next state.
        Actions:
        0 = idle
        1 = move
        2 = execute task
        3+i = select task i
        """
        
        def choosing_new_target(task_choice):
            self.target = task_choice
            task_choice.selected = True
            task_choice.assignee = self
            self.state = STATAgent.MOVING
            self.move()

            if self.at_target():
                self.state_time_counter = self.model.global_time_to_execute_RL - 1
                reward = pos_rew + 2
                return reward
            elif get_distance(self.pos, self.target.pos) < close_factor:
                reward = pos_rew + 1.5
                return reward
            else:
                reward = pos_rew
                return reward
        
        time_penalty = 0.5 * (self.model.numSteps // 10)
        reward = 0
        stable_rew = -1
        base_reward = 30 - time_penalty  
        close_factor = 1.0
        pos_rew = 5.0 - time_penalty
        neg_rew = -1.5 - time_penalty 
        inv_rew = -3.0 - time_penalty

        if action == 0:  # Idle
            if (all(task.completed == 1 for task in self.model.sorted_tasks) or all(task.selected == 1 for task in self.model.sorted_tasks)) and self.target is None:
                self.state = STATAgent.IDLE
                reward = 0
            else:
                reward = stable_rew
                
            
        elif action == 1:  # Move
            if self.state == STATAgent.IDLE and self.target and (self.target.completed == 0):
                self.state = STATAgent.MOVING
                self.move()
                if self.at_target():
                    self.state_time_counter = self.model.global_time_to_execute_RL - 1
                reward = stable_rew

            elif self.state == STATAgent.MOVING and self.target and (self.target.completed == 0):
                if self.at_target():
                    reward = neg_rew
                else:
                    moved = self.move()
                    if moved:
                        if self.at_target():
                            self.state_time_counter = self.model.global_time_to_execute_RL - 1
                        reward = stable_rew
                    else:
                        reward = neg_rew
            else:
                reward = neg_rew
            
        elif action == 2:  # Execute task
            if self.state == STATAgent.MOVING and self.at_target() and self.target.completed == 0:
                self.state = STATAgent.EXECUTING
                self.state_time_counter -= 1
                reward = stable_rew

            elif self.state == STATAgent.EXECUTING and self.at_target() and self.target.completed == 0:
                if self.state_time_counter == 0:
                    self.execute_task(self.target)
                    self.target = None
                    self.state = STATAgent.IDLE
                    self.state_time_counter = self.model.global_time_to_execute_RL
                    reward = base_reward * (1 + 0.1 * self.model.total_tasks_completed)
                else:
                    self.state_time_counter -= 1
                    reward = stable_rew

            else:
                reward = neg_rew
                
                
        elif action >= 3:  # Select task i.
            if self.state == STATAgent.EXECUTING and self.target and self.target.completed == 0:
                reward = inv_rew
                if DEBUG:
                    print("[DEBUG] Invalid: agent is executing a task and selected a new task.")
            else:
                task_idx = action - 3  

                if 0 <= task_idx < len(self.model.sorted_tasks):
                    task_choice = self.model.sorted_tasks[task_idx]

                    if (self.state == STATAgent.IDLE or self.state == STATAgent.MOVING) and task_choice.completed == 0:
                        if task_choice.selected == False:
                            reward = choosing_new_target(task_choice)
                            reward = stable_rew
                        else:
                            if get_distance(task_choice.assignee.pos, task_choice.pos) < close_factor:
                                reward = choosing_new_target(task_choice)
                                reward = stable_rew 
                                if DEBUG:
                                    print("[DEBUG] Reassigned task from a nearby agent.")
                            else:
                                task_choice.assignee.target = None
                                task_choice.assignee.state = STATAgent.IDLE
                                task_choice.assignee.state_time_counter = self.model.global_time_to_execute_RL
                                reward = choosing_new_target(task_choice)
                                reward = stable_rew
                                if DEBUG:
                                    print("[DEBUG] Reassigned task from another agent.")

                    else:
                        reward = inv_rew
                        if DEBUG:
                            print("[DEBUG] Invalid select-task action: task is already completed or agent state is invalid.")

                else:
                    reward = inv_rew
                    if DEBUG:
                        print("[DEBUG] Invalid select-task action: task index is out of range.")
       
        return action, reward

    def at_target(self) -> bool:
        if self.target:
            return self.target and self.target.pos and get_distance(self.pos, self.target.pos) < 0.5
        else:
            return False
    
    
    def move(self) -> bool:
        """
        Select a target if necessary and move toward it.

        Returns true if there is a task to visit, false if all have been completed.
        """
        if self.policy not in [5,6]:
            if not self.find_target():
                return False
        target = self.target
        if target:
            if not target.pos:
                raise ValueError("ERROR: Target has no position")

            new_pos, delta_x, delta_y = calc_move(*self.pos, *target.pos, self.speed)
            self.model.space.move_agent(self, new_pos)
            return True
        else:
            return False
  
    def execute_task(self, target):
        if target.completed == 0:
            target.completed = 1
            self.num_completed += 1
            self.model.total_tasks_completed += 1
            self.model.total_tasks_completed_now += 1


class STATTask(mesa.Agent):
    """Stationary STAT task.

    Domain-general role: a spatial task that can be selected and completed.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.completed = 0
        self.health = random.uniform(0, 1)
        self.selected = False
        self.assignee = None



class STATModel(mesa.Model):
    """STAT core model with mobile agents, stationary tasks, and discrete time."""
    def __init__(self, seed, agents, tasks, width, height, policy, num_bins, agent_speeds=None):
        super().__init__()
        
        self.seed_val = seed
        random.seed(seed)
        np.random.seed(seed)

        self.num_agents = agents
        self.num_tasks = tasks
        self.policy = policy
        self.space = ContinuousSpace(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.width = width
        self.height = height
        self.numSteps = 0
        self.global_time_to_execute_RL = 3
        self.all_tasks_completed = False
        self.num_bins = num_bins
        self.agent_speeds = agent_speeds or [1.0] * self.num_agents

        self.add_agents()
        self.add_tasks()

        self.num_sub_actions = 3 + self.num_tasks

        
        self.total_tasks_completed = 0
        self.total_tasks_completed_now = 0
        
        
        self.datacollector = mesa.DataCollector(
            model_reporters={"Tasks Completed": compute_completed, "Steps Taken": lambda m: m.numSteps},
            agent_reporters={"Task Completed": "completed", "Position": "pos", "Health": "health", "State": "state"},
        )
        
        
    def prep_q_learning(self):
        """Prepare state/action structures for legacy tabular Q-learning."""
        if self.num_bins < 2:
            self.num_bins = 2
        
        self.global_time_to_execute_RL = 3
        self.global_state_space_size = (
            (self.num_bins**(self.num_tasks*self.num_agents))
            * (3 ** self.num_agents)
            * (2**self.num_tasks)
            * (2**self.num_tasks)
        )

        self.num_sub_actions = 3 + self.num_tasks
        self.all_joint_actions = sorted(list(itertools.product(range(self.num_sub_actions), repeat=self.num_agents)))
        self.global_action_space_size = len(self.all_joint_actions)


            
    def add_agents(self):     
        self.sorted_agents = []

        for i in range(self.num_agents):
            speed = self.agent_speeds[i] if hasattr(self, "agent_speeds") else 1.0
            r = STATAgent(i, self, self.policy, speed=speed)
            self.schedule.add(r)
            self.space.place_agent(r, (0, 0))
            self.sorted_agents.append(r)

            if DEBUG:
                print(f"Agent {i} added, speed is {r.speed}")
                print(f"[DEBUG] Total agents in list: {len(self.sorted_agents)}")

            
    def add_tasks(self):
        self.sorted_tasks = []

        for j in range(self.num_agents, self.num_agents + self.num_tasks):
            task = STATTask(j, self)
            self.schedule.add(task)
            x = random.uniform(0, self.space.width)
            y = random.uniform(0, self.space.height)
            self.space.place_agent(task, (x, y))
            self.sorted_tasks.append(task)
        
    def train_qlearning(self, num_episodes, num_bins, learning_rate, discount_factor, epsilon_value, epsilon_decay):
        """
        Train the Q-learning policy.
        """
        self.global_q_table = {}
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_value
        self.max_epsilon = 1.0
        self.min_epsilon = 0
        self.decay_rate = epsilon_decay
        self.episodes = num_episodes
        
        episodes = []
        rewards = []
        steps = []
        epsilon_values = []
        
        max_steps = 50
        
        for episode in range(num_episodes):
            if episode != 0:
                self.reset()
        
            self.episode_total_reward = 0
            self.episode_action_history = []

            while not all(task.completed == 1 for task in self.sorted_tasks):
                self.q_learning_step()
   
            if len(rewards) > 0:
                avg_reward = sum(rewards) / len(rewards)
                adjusted_reward = max(0, avg_reward)
                self.epsilon = max(self.min_epsilon, self.epsilon * (1 - self.decay_rate * (1 + adjusted_reward / 100)))
            else:
                self.epsilon = max(self.min_epsilon, self.epsilon * (1 - self.decay_rate))

            print(f"Episode {episode + 1}/{num_episodes}: Total reward = {self.episode_total_reward}, Epsilon: {self.epsilon}")
            
            if all(task.completed == 1 for task in self.sorted_tasks):
                print(f"All tasks completed in {self.numSteps} steps!")
            else:
                print(f"All tasks NOT COMPLETED in {self.numSteps} steps ):")
                
            print(f"Epsilon: {self.epsilon:.4f}")

            
            episodes.append(episode + 1)
            rewards.append(self.episode_total_reward)
            steps.append(self.numSteps)
            epsilon_values.append(self.epsilon)

        print("Training complete!")   
        return episodes, rewards, steps, epsilon_values
        
        
    def get_global_state(self):
        """
        Compute the global current state as a tuple for Q-learning
        """
        def bin_distance(distance, width, num_bins):
            if distance < 0.5:
                return 0
            elif distance < 1.0: 
                return 1
            else:
                return min(int(distance // (self.space.width / self.num_bins)), self.num_bins - 1)
        
        binned_distances = [
            bin_distance(get_distance(agent.pos, task.pos), self.space.width, self.num_bins)
            for agent in self.sorted_agents
            for task in self.sorted_tasks
        ]
        agent_states = [agent.state for agent in self.sorted_agents]
        task_status = [task.completed for task in self.sorted_tasks]
        task_selected = [1 if task.selected else 0 for task in self.sorted_tasks]

        global_state = tuple(
            binned_distances + agent_states + task_status + task_selected
        )
        return global_state

        
    
    def reset(self, seed=None):
        """
        Reset the model to the initial state for a new episode.
        """
        if seed is not None:
            self.seed_val = seed
            random.seed(self.seed_val)
            np.random.seed(self.seed_val)

        self.space = ContinuousSpace(self.width, self.height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)
        self.numSteps = 0
        self.all_tasks_completed = False

        self.add_agents()
        self.add_tasks()
        self.total_tasks_completed = 0
        self.total_tasks_completed_now = 0
    
    def encode_state(self, state):
        """Encodes a state into a hashable index for dictionary-based Q-table."""
        state_tuple = tuple(state)

        if state_tuple in self.global_q_table:
            return self.global_q_table[state_tuple]

        new_state_idx = len(self.global_q_table)
        self.global_q_table[state_tuple] = new_state_idx
        return new_state_idx



    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.numSteps += 1
