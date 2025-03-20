#import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
from Actor_Critic_Train import actor_critic_func
import pickle
from matplotlib import pyplot as plt
# This environment allows you to verify whether your program runs correctly during testing,
# as it follows the same observation format from `env.reset()` and `env.step()`.
# However, keep in mind that this is just a simplified environment.
# The full specifications for the real testing environment can be found in the provided spec.
#
# You are free to modify this file to better match the real environment and train your own agent.
# Good luck!


class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=50):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False

        self.stations = []
        for i in range(4):
            while True:
                station=(random.randint(0, self.grid_size - 1),random.randint(0, self.grid_size - 1))
                if station not in self.stations:
                    self.stations.append(station)
                    break
                else:
                    continue
        self.passenger_loc = None
        if grid_size>=8:
            obstacle_num=random.randint(8,20)
        elif grid_size==5:
            obstacle_num=random.randint(0,5)
        else:
            obstacle_num=random.randint(3,10)
        self.obstacles = []
        for i in range(obstacle_num):
            while True:
                obstacle=(random.randint(0, self.grid_size - 1),random.randint(0, self.grid_size - 1))
                if obstacle not in self.stations and obstacle not in self.obstacles:
                    self.obstacles.append(obstacle)
                    break
                else:
                    continue
        # No obstacles in simple version
        
        self.action_history=[]
        self.epsilon=0.75
        self.decay=0.9995
        self.check_point1=False
        self.check_point2=False
        self.check_point3=False
        self.check_point4=False
        self.check_point5=False
        self.check_point6=False
        self.check_point7=False
        self.check_point8=False
        self.passenger_loc_num=0

    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        self.stations = []
        for i in range(4):
            while True:
                station=(random.randint(0, self.grid_size - 1),random.randint(0, self.grid_size - 1))
                if station not in self.stations:
                    self.stations.append(station)
                    break
                else:
                    continue
        self.passenger_loc = None
        if self.grid_size>=8:
            obstacle_num=random.randint(8,20)
        elif self.grid_size==5:
            obstacle_num=random.randint(0,5)
        else:
            obstacle_num=random.randint(3,10)
        self.obstacles = []
        for i in range(obstacle_num):
            while True:
                obstacle=(random.randint(0, self.grid_size - 1),random.randint(0, self.grid_size - 1))
                if obstacle not in self.stations and obstacle not in self.obstacles:
                    self.obstacles.append(obstacle)
                    break
                else:
                    continue

        available_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.stations and (x, y) not in self.obstacles
        ]

        self.taxi_pos = random.choice(available_positions)

        self.passenger_loc = random.choice([pos for pos in self.stations])
        for i in range(4):
            if self.passenger_loc==self.stations[i]:
                self.passenger_loc_num=i

        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)

        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        pre_dis1=abs(taxi_row-self.stations[0][0])+abs(taxi_col-self.stations[0][1])
        dis1=abs(next_row-self.stations[0][0])+abs(next_col-self.stations[0][1])
        pre_dis2=abs(taxi_row-self.stations[1][0])+abs(taxi_col-self.stations[1][1])
        dis2=abs(next_row-self.stations[1][0])+abs(next_col-self.stations[1][1])
        pre_dis3=abs(taxi_row-self.stations[2][0])+abs(taxi_col-self.stations[2][1])
        dis3=abs(next_row-self.stations[2][0])+abs(next_col-self.stations[2][1])
        pre_dis4=abs(taxi_row-self.stations[3][0])+abs(taxi_col-self.stations[3][1])
        dis4=abs(next_row-self.stations[3][0])+abs(next_col-self.stations[3][1])
        if len(self.action_history)==8:
            self.action_history.pop(0)
        self.action_history.append(action)
        
        rand_num=random.random()
        
        def find_cycle(arr):
            n = len(arr)
            
            # # 如果所有元素都相同，直接返回 False，因為這不是循環
            # if len(set(arr)) == 1:
            #     return False
            
            # 嘗試每種循環長度
            for cycle_len in range(2, n // 2 + 1):  # 至少 2，最多 n//2
                if n % cycle_len == 0:  
                    pattern = arr[:cycle_len]
                    # 檢查是否所有段都是相同的 pattern
                    if all(arr[i:i+cycle_len] == pattern for i in range(0, n, cycle_len)):
                        return True 
                        
            return False  # 沒有發現循環
        if rand_num<self.epsilon:
            if find_cycle(self.action_history):
                reward-=0.3
            self.epsilon=max(self.epsilon*self.decay,0.3)

        passenger_loc_north = int((next_row - 1, next_col) == self.passenger_loc)
        passenger_loc_south = int((next_row + 1, next_col) == self.passenger_loc)
        passenger_loc_east  = int((next_row, next_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((next_row, next_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (next_row, next_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle

        destination_loc_north = int( (next_row - 1, next_col) == self.destination)
        destination_loc_south = int( (next_row + 1, next_col) == self.destination)
        destination_loc_east  = int( (next_row, next_col + 1) == self.destination)
        destination_loc_west  = int( (next_row, next_col - 1) == self.destination)
        destination_loc_middle  = int( (next_row, next_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -=6
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
            if (destination_look or passenger_look) and self.passenger_picked_up:
                reward+=0.18
            if self.taxi_pos == self.destination:
                reward += 0.2
            if self.taxi_pos == self.passenger_loc:
                reward+=0.2
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc and self.passenger_picked_up==False:
                    self.passenger_picked_up = True
                    reward+=15
                    self.passenger_loc = self.taxi_pos
                elif passenger_look and self.passenger_picked_up==False:
                    pass
                elif self.passenger_picked_up:
                    reward-=20
                else:
                    reward = -15
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 10000
                        return self.get_state(), reward -0.1, True, {}
                    elif destination_look:
                        pass
                    else:
                        reward -=15
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -=20
        if self.taxi_pos == self.destination and action!=5:
            reward -= 0.1
        if self.taxi_pos == self.passenger_loc and self.passenger_picked_up==False and action!=4:
            reward -=0.1
        # if self.taxi_pos == self.stations[0]:
        #     if self.taxi_pos==self.passenger_loc and self.passenger_picked_up:
        #         self.check_point1=True
        #         self.check_point5=True
        #     if self.passenger_picked_up:
        #         self.check_point5=True
        #     if self.taxi_pos!=self.passenger_loc:
        #         self.check_point1=True
        # if self.taxi_pos == self.stations[1]:
        #     if self.taxi_pos==self.passenger_loc and self.passenger_picked_up:
        #         self.check_point2=True
        #         self.check_point6=True
        #     if self.passenger_picked_up:
        #         self.check_point6=True
        #     if self.taxi_pos!=self.passenger_loc:
        #         self.check_point2=True
        # if self.taxi_pos == self.stations[2]:
        #     if self.taxi_pos==self.passenger_loc and self.passenger_picked_up:
        #         self.check_point3=True
        #         self.check_point7=True
        #     if self.passenger_picked_up:
        #         self.check_point7=True
        #     if self.taxi_pos!=self.passenger_loc:
        #         self.check_point3=True
        # if self.taxi_pos == self.stations[3]:
        #     if self.taxi_pos==self.passenger_loc and self.passenger_picked_up:
        #         self.check_point8=True
        #         self.check_point4=True
        #     if self.passenger_picked_up:
        #         self.check_point8=True
        #     if self.taxi_pos!=self.passenger_loc:
        #         self.check_point4=True
        # if self.passenger_picked_up==False:
        #     if self.check_point1==False:
        #         if dis1<pre_dis1:
        #             reward+=0.15
        #     elif self.check_point1==True and self.check_point2==False:
        #         if dis2<pre_dis2:
        #             reward+=0.15
        #     elif self.check_point1==True and self.check_point2==True and self.check_point3==False:
        #         if dis3<pre_dis3:
        #             reward+=0.15
        #     elif self.check_point1==True and self.check_point2==True and self.check_point3==True and self.check_point3==False:
        #         if dis4<pre_dis4:
        #             reward+=0.15
        # if self.passenger_picked_up==True:
        #     if self.check_point5==False:
        #         if dis1<pre_dis1:
        #             reward+=0.15
        #     elif self.check_point5==True and self.check_point6==False:
        #         if dis2<pre_dis2:
        #             reward+=0.15
        #     elif self.check_point5==True and self.check_point6==True and self.check_point7==False:
        #         if dis3<pre_dis3:
        #             reward+=0.15
        #     elif self.check_point5==True and self.check_point6==True and self.check_point7==True and self.check_point8==False:
        #         if dis4<pre_dis4:
        #             reward+=0.15
        reward -= 0.1

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -100, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination

        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle

        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle


        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]


        # Place passenger
        py, px = self.passenger_loc
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'



        grid[self.stations[0][0]][self.stations[0][1]]='R'
        grid[self.stations[1][0]][self.stations[1][1]]='G'
        grid[self.stations[2][0]][self.stations[2][1]]='Y'
        grid[self.stations[3][0]][self.stations[3][1]]='B'

        # Place destination
        dy, dx = self.destination
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'
        #Place Obstacles
        for ox,oy in self.obstacles:
            grid[ox][oy] = '|'
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        #print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        #print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")
        print(f"Passenger in taxi: {self.passenger_picked_up}")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:


        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward


def train_agent(agent_file, grid_size,fuel_limit, episodes=5000,ac=None):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    #spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(grid_size=grid_size, fuel_limit=fuel_limit)
    obs, _ = env.reset()
    action=[0, 1, 2, 3, 4, 5]
    if ac==None:
        actor_critic=actor_critic_func(obs,action)
    else:
        actor_critic=ac
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    decay_rate=0.9999
    alpha=8e-4
    beta=5e-3
    minimum_lr_a=4e-4
    minimum_lr_b=1e-3
    nstep=100
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    rewards_per_episode=[]
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        actor_critic.trajectory=[]
        mid_q_state=None
        while not done:

            actor_critic.make_new_state(obs)
            action = actor_critic.get_action(obs)

            next_obs, reward, done, _ = env.step(action)
            if (step_count)  == 2000:
                mid_q_state=actor_critic.get_key_state(obs)
#             if step_count>3000 and episode>800:
#                 print(f"reward: {reward},step:{step_count}, 1:{env.check_point1},2:{env.check_point2},3:{env.check_point3}\
# , 4:{env.check_point4},5:{env.check_point5},6:{env.check_point6},7:{env.check_point7},8:{env.check_point8},\
# passenger:{env.passenger_picked_up},pass_loc:{env.passenger_loc_num} {env.passenger_loc},stat1:{env.stations[1]}, taxi_pos:{env.taxi_pos}, ")
#                 time.sleep(0.2)
            total_reward += reward
            actor_critic.append_trajectory(obs,action,reward,nstep)
            actor_critic.make_new_state(next_obs)
            actor_critic.n_step_update(actor_critic.trajectory,next_obs,nstep=nstep,alpha=alpha,beta=beta)
            if done:
                actor_critic.final_update(actor_critic.trajectory,next_obs,alpha=alpha,beta=beta)
            step_count += 1
            obs=next_obs
            taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
    
        alpha*=decay_rate
        beta*=decay_rate
        alpha=max(alpha,minimum_lr_a)
        beta=max(beta,minimum_lr_b)
        rewards_per_episode.append(total_reward)
        state=actor_critic.get_key_state(obs)
        if reward>50:
            print(f"episode:{episode},time_step:{step_count},state:{state}, q table : {actor_critic.q_table[state]}")

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}")
            
            if mid_q_state:
                print(f"state:{mid_q_state}, mid q table : {actor_critic.q_table[mid_q_state]}")
            print(f"state:{state},q table : {actor_critic.q_table[state]},time_step:{step_count}")
            print(f"alpha:{alpha}, beta:{beta}")
            actor_critic.count_keys()
    print(actor_critic.len_keys)
    return actor_critic,rewards_per_episode

if __name__ == "__main__":

    AC,rewards=train_agent("student_agent.py",grid_size=10 ,fuel_limit=5000,episodes=20000,ac=None)
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Tabular Policy Learning Training Progress")
    plt.savefig(f'rewards_per_episode.png')
    with open('train_q_table.pkl','wb')as f:
        pickle.dump(AC.q_table,f) 
    # agent_score = run_agent("student_agent.py", env_config, render=False)
    # print(f"Final Score: {agent_score}")