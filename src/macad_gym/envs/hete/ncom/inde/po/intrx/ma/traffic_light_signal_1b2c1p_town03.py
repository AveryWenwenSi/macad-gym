#!/usr/bin/env python
import time

from macad_gym.carla.multi_env import MultiCarlaEnv


class TrafficLightSignal1B2C1PTown03(MultiCarlaEnv):
    """A 4-way signalized intersection with 1 Bike, 2 Cars, 1 Pedestrian"""
    def __init__(self):
        self.configs = {
            "scenarios": "DEFAULT_SCENARIO_TOWN1",
            "env": {
                "server_map": "/Game/Carla/Maps/Town01",
                "render": True,
                "render_x_res": 800,
                "render_y_res": 600,
                "x_res": 400,
                "y_res": 400,
                "framestack": 1,
                "discrete_actions": True,
                "squash_action_logits": False,
                "verbose": False,
                "use_depth_camera": False,
                "send_measurements": False,
                "enable_planner": True,
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
            },
            "actors": {
                "vehicle1": {
                    # to be trained
                    "type": "vehicle_4W",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "flag_v1",
                    "manual_control": True,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": True,
                    "x_res": 84,
                    "y_res": 84,
                    "use_depth_camera": False,
                    "send_measurements": False,
                },
                "vehicle2": {
                    # autopilot
                    "type": "vehicle_4w",
                    "enable_planner": False,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "flag_v2",
                    "manual_control": False,
                    "auto_control": True,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 84,
                    "y_res": 84,
                    "use_depth_camera": False,
                    "send_measurements": False,
                },
            },
        }
        super(TrafficLightSignal1B2C1PTown03, self).__init__(self.configs)


if __name__ == "__main__":
    env = TrafficLightSignal1B2C1PTown03()
    configs = env.configs
    for ep in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        env_config = configs["env"]
        actor_configs = configs["actors"]
        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env._discrete_actions:
                action_dict[actor_id] = 3  # Forward
            else:
                action_dict[actor_id] = [1, 0]  # test values

        start = time.time()
        i = 0
        done = {"__all__": False}
        while not done["__all__"]:
            # while i < 20:  # TEST
            i += 1
            obs, reward, done, info = env.step(action_dict)
            # action_dict = get_next_actions(info, env.discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew",
                                  "done{}"]).format(i, reward,
                                                    total_reward_dict, done))

            time.sleep(0.1)

        print("{} fps".format(i / (time.time() - start)))
