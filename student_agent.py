# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
# import gym
with open('train_q_table.pkl','rb') as f:
    q=pickle.load(f)
def softmax(x):
    """✅ Compute softmax values for an array."""
    exp_x = np.exp(x - np.max(x))  # Numeric stability
    return exp_x / exp_x.sum()
def get_key_state(state):
    key_state=f"({state[0]},{state[1]})_({state[2]},{state[3]})_({state[4]},{state[5]})_({state[6]},{state[7]})_({state[8]},{state[9]})_({state[10]},{state[11]},{state[12]},{state[13]})_{state[14]}_{state[15]}"
    return key_state
def get_action(obs):
    initial_state=get_key_state(obs)
    for i in range(800):
        # TODO: Train your own agent
        key_state=get_key_state(obs)
        if key_state not in q.keys():
            taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
            obs=[taxi_row, taxi_col,random.randint(0,9),random.randint(0,9),random.randint(0,9),random.randint(0,9),\
                random.randint(0,9),random.randint(0,9),random.randint(0,9),random.randint(0,9),obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look]
        else:
            q[initial_state]=q[key_state]
            action_probs = softmax(q[key_state])
            action = np.random.choice(6, p=action_probs)
            return action # Choose a random action
    return random.choice([0,1,2,3,4,5])
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    

    
    # You can submit this random agent to evaluate the performance of a purely random strategy.
if __name__=="__main__":
    print(q)
