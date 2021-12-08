import argparse
import numpy as np
import torch

import multiprocessing as mp

from core import *
from manager import *
from process import *

# default_config_path = "config.YOUR_AGENT.YOUR_ENV"
default_config_path = "config.qmix.smac"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config.qmix.smac")
    args, unknown = parser.parse_known_args()
    config_path = args.config if args.config else default_config_path
    config_manager = ConfigManager(config_path, unknown)
    config = config_manager.config

    env_class = Env(**config.env)
    env = env_class.env
    env_info = env.get_env_info()
    agent_config = {
        "num_agents": env_info["n_agents"],
        "shape_obs": env_info["obs_shape"] + env_info["n_agents"],
        "shape_state": env_info["state_shape"],
        "num_actions_set": [env_info["n_actions"]],
        #"state_size": env.state_size,
        #"action_size": env.action_size,
        "optim_config": config.optim,
        "run_step": config.train.run_step,
    }
    agent_config.update(config.agent)

    result_queue = mp.Queue()
    manage_sync_queue = mp.Queue(1)
    path_queue = mp.Queue(1)

    record_period = (
        config.train.record_period
        if config.train.record_period
        else config.train.run_step // 10
    )
    eval_manager_config = (
        Env,
        config.env,
        config.train.eval_iteration,
        config.train.record,
        record_period,
    )
    log_id = config.train.id if config.train.id else config.agent.name
    log_manager_config = (config.env.name, log_id, config.train.experiment)
    manage = mp.Process(
        target=manage_process,
        args=(
            Agent,
            {"device": "cpu", **agent_config},
            result_queue,
            manage_sync_queue,
            path_queue,
            config.train.run_step,
            config.train.print_period,
            MetricManager,
            EvalManager,
            eval_manager_config,
            LogManager,
            log_manager_config,
            config_manager,
        ),
    )
    manage.start()
    try:
        agent = Agent(**agent_config)
        assert agent.action_type == env_class.action_type
        if config.train.load_path:
            agent.load(config.train.load_path)

        save_path = path_queue.get()
        """
        for step in range(1, config.train.run_step + 1):
            action_dict = agent.act(state, config.train.training)
            next_state, reward, done = env.step(action_dict["action"])
            transition = {
                "state": state,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }
            transition.update(action_dict)
            transition = agent.interact_callback(transition)
            if transition:
                result = agent.process([transition], step)
                result_queue.put((step, result))
            if step % config.train.print_period == 0 or step == config.train.run_step:
                try:
                    manage_sync_queue.get_nowait()
                except:
                    pass
                manage_sync_queue.put(agent.sync_out())
            if step % config.train.save_period == 0 or step == config.train.run_step:
                agent.save(save_path)

            state = next_state if not done else env.reset()
            """
        step_cnt = 0
        done_cnt = 0
        obs_0_idx = np.eye(env_info["n_agents"])
        for epi_cnt in range(config.train.max_episode):
            env.reset()
            episode_reward = 0
            actions_last = env.last_action
            hidden_last = np.zeros((env_info["n_agents"], 64))
            agent.memory.create_new_episode()
            for epi_step_cnt in range(1, config.train.run_step + 1):
                step_cnt += 1 # update the cnt every time
  
                # get obs state for select action
                state = env.get_state()
                obs = np.concatenate([obs_0_idx, np.array(env.get_obs())], axis=1)
                avail_actions = np.array(env.get_avail_actions())
  
                # interact with the env and get new state obs
                actions, hidden = agent.act(avail_actions, obs, actions_last, hidden_last, args)
                reward, done, _ = env.step(actions)
                reward = reward*config.train.reward_scale_par # normalize the reward
                if epi_step_cnt == config.train.per_episode_max_len-1: done = True # max len of   episode
                state_new = env.get_state()
                obs_new = np.concatenate([obs_0_idx, np.array(env.get_obs())], axis=1)
                avail_actions_new = np.array(env.get_avail_actions())
                actions_now_onehot = env.last_action # the env do the things for us
  
                # update the date and save experience to memory
                if done == True: done_cnt += 1
  
                #concatenate the obs and actions_last for speed up the train
                agent.save_memory(np.concatenate([obs, actions_last], axis=-1), state, actions.reshape(1, -1), avail_actions_new, np.concatenate([obs_new, actions_now_onehot], axis=-1), state_new, reward, done)
                """
                transition={
                        "obs" : obs,
                        "actions_last" : actions_last,
                        "state": state,
                        "actions_reshaped" : actions.reshape(1,-1),
                        "avail_actions_new" : avail_actions_new,
                        "obs_new" : obs_new,
                        "actions_now_onehot" : actions_now_onehot,
                        "state_new" : state_new,
                        "reward" : reward,
                        "done" : done,
                }
                """
                actions_last = env.last_action
                hidden_last = hidden
  
                # agents learn
                #loss = agent.learn(step_cnt, epi_cnt)
                #print(' '*80, 'loss is', loss, end='\r')
                #print(f"episode : {epi_cnt} , step_cnt : {step_cnt}, loss : {loss} , reward : {episode_reward} , epsilon : {agent.epsilon},  \n")

                # if done, end the episode
                episode_reward += reward
                if agent.epsilon > agent.epsilon_min :
                    agent.epsilon -= agent.anneal_par
                if done: break

            loss = agent.learn(step_cnt, epi_cnt)
            print(f"episode : {epi_cnt} , step_cnt : {step_cnt}, loss : {loss} , reward : {episode_reward} , epsilon : {agent.epsilon},  \n")

    except Exception as e:
        traceback.print_exc()
        manage.terminate()
    else:
        print("Optimize process done.")
        manage.join()
        print("Manage process done.")
    finally:
        result_queue.close()
        manage_sync_queue.close()
        path_queue.close()
        env.close()
