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
    key_state=f"({state[0]},{state[1]})_({state[10]},{state[11]},{state[12]},{state[13]})_{state[14]}_{state[15]}"
    #key_state=f"({state[0]},{state[1]})_({state[10]},{state[11]},{state[12]},{state[13]})_{state[14]}_{state[15]}"
    return key_state
def get_action(obs):

    key_state=get_key_state(obs)
    if key_state not in q.keys() or not (obs[10] or obs[11] or obs[12] or obs[13] or obs[14] or obs[15]):
        action = random.choice([0,1,2,3,4,5])
    else:
        action_probs = softmax(q[key_state])
        action = np.random.choice(6, p=action_probs)
    return action
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    

    
    # You can submit this random agent to evaluate the performance of a purely random strategy.
if __name__=="__main__":
    print(q)
