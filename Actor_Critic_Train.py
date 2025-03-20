import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
class actor_critic_func():
  def __init__(self,state,action,lr=0.1):
    self.state=state
    self.action=action
    self.q_table={}
    self.v_table={}
    self.trajectory=[]
    self.len_keys=[]
  def count_keys(self):
    self.len_keys.append(len(self.q_table.keys()))
  def softmax(self,x):
    """✅ Compute softmax values for an array."""
    exp_x = np.exp(x - np.max(x))  # Numeric stability
    return exp_x / exp_x.sum()
  def get_key_state(self,state):
    key_state=f"({state[0]},{state[1]})_({state[10]},{state[11]},{state[12]},{state[13]})_{state[14]}_{state[15]}"
    return key_state
  def make_new_state(self,state):
    key_state=self.get_key_state(state)
    if key_state not in self.q_table.keys():
      self.q_table[key_state]=np.ones(len(self.action))
    if key_state not in self.v_table.keys():
      self.v_table[key_state]=0
  def append_trajectory(self,state,action,reward,nstep):
    if len(self.trajectory)==nstep:
      self.trajectory.pop(0)
    state=self.get_key_state(state)
    self.trajectory.append((state,action,reward))
  def get_action(self,state):
    key_state=self.get_key_state(state)
    action_probs = self.softmax(self.q_table[key_state])
    action = np.random.choice(len(self.action), p=action_probs)
    return(action)
  def n_step_update(self,trajectory,next_state,nstep=5,alpha=0.1,beta=0.1,gamma=0.98)->None:
    next_state=self.get_key_state(next_state)
    G=0
    if len(self.trajectory)==nstep:
      for state,action,reward in reversed(self.trajectory):
        G=reward+gamma*G
        state=state
        action=action
      delta = G - self.v_table[state]
      #actor policy update
      action_probs=self.softmax(self.q_table[state])
      for a in range(len(self.action)):
        if a == action:
          self.q_table[state][action] += alpha*delta*(1-action_probs[a])*action_probs[a]
        else:
          self.q_table[state][a] += alpha*delta*(0-action_probs[a])*action_probs[a]

      #value update
      self.v_table[state]+=beta*delta
    

  def final_update(self,trajectory,next_state,alpha=0.01,beta=0.01,gamma=0.98)->None:
    next_state=self.get_key_state(next_state)
    G=0
    for state,action,reward in reversed(self.trajectory):
      G=reward+gamma*G
      state=state
      action=action
      delta = G  - self.v_table[state]
      #actor policy update
      action_probs=self.softmax(self.q_table[state])
      for a in range(len(self.action)):
        if a == action:
          self.q_table[state][action] += alpha*delta*(1-action_probs[a])*action_probs[a]
        else:
          self.q_table[state][a] += alpha*delta*(0-action_probs[a])*action_probs[a]

      #value update
      self.v_table[state]+=beta*delta

