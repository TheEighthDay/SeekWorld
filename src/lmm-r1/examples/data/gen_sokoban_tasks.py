import gym  
import gym_sokoban  
import numpy as np  
import os  
from PIL import Image  
from tqdm import tqdm


def save_level_and_initial_frame(env, level_index, env_name):  
    # Save the generated level (map)  
    level_data = env.unwrapped.room_state  
    np.save(f'sokoban_levels/{env_name}-level_{level_index}.npy', level_data)  
  
    # Save the initial frame as an image  
    initial_frame = env.render(mode='rgb_array')  
    img = Image.fromarray(initial_frame)
    w,h = img.size
    new_h = 640
    new_w = int(new_h/h*w)
    img = img.resize((new_w, new_h))
    img.save(f'sokoban_initial_frames/{env_name}-frame_{level_index}.png')  
  
def main():  
    env_name = 'Sokoban-small-v0'  
    num_tasks = 5000
    os.makedirs('sokoban_levels', exist_ok=True)  
    os.makedirs('sokoban_initial_frames', exist_ok=True)  
    
    for i in tqdm(range(num_tasks)):  
        env = gym.make(env_name)  
        env.reset()  
        save_level_and_initial_frame(env, i, env_name)  
        env.close()  
    
    env_name = 'Sokoban-small-v1'  
    num_tasks = 5000
    os.makedirs('sokoban_levels', exist_ok=True)  
    os.makedirs('sokoban_initial_frames', exist_ok=True)  
    
    for i in tqdm(range(num_tasks)):  
        env = gym.make(env_name)  
        env.reset()  
        save_level_and_initial_frame(env, i, env_name)  
        env.close()
  
if __name__ == "__main__":  
    main()  