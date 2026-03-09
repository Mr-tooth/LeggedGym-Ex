import time

from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick

try:
    import pygame
except ImportError:
    pygame = None


class KeyboardControl:
    def __init__(self, cmd_step=0.05, yaw_step=0.1):
        if pygame is None:
            raise ImportError("pygame is required for keyboard control")
        pygame.init()
        self.screen = pygame.display.set_mode((520, 120))
        pygame.display.set_caption("LeggedGym Keyboard Control")
        self.font = pygame.font.Font(None, 22)
        self.cmd_step = cmd_step
        self.yaw_step = yaw_step
        self.cmd_x = 0.0
        self.cmd_y = 0.0
        self.cmd_yaw = 0.0

    def update(self, env):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            self.cmd_x += self.cmd_step
        if keys[pygame.K_s]:
            self.cmd_x -= self.cmd_step
        if keys[pygame.K_d]:
            self.cmd_y += self.cmd_step
        if keys[pygame.K_a]:
            self.cmd_y -= self.cmd_step
        if keys[pygame.K_q]:
            self.cmd_yaw += self.yaw_step
        if keys[pygame.K_e]:
            self.cmd_yaw -= self.yaw_step

        if keys[pygame.K_r]:
            self.cmd_x = 0.0
            self.cmd_y = 0.0
            self.cmd_yaw = 0.0

        self.cmd_x = float(np.clip(self.cmd_x, env.command_ranges["lin_vel_x"][0], env.command_ranges["lin_vel_x"][1]))
        self.cmd_y = float(np.clip(self.cmd_y, env.command_ranges["lin_vel_y"][0], env.command_ranges["lin_vel_y"][1]))
        self.cmd_yaw = float(np.clip(self.cmd_yaw, env.command_ranges["ang_vel_yaw"][0], env.command_ranges["ang_vel_yaw"][1]))

        env.commands[:, 0] = self.cmd_x
        env.commands[:, 1] = self.cmd_y
        env.commands[:, 2] = self.cmd_yaw
        env.commands[:, 3] = 0.0

        text = f"W/S: x {self.cmd_x:+.2f}  A/D: y {self.cmd_y:+.2f}  Q/E: yaw {self.cmd_yaw:+.2f}  R: reset"
        self.screen.fill((25, 25, 25))
        self.screen.blit(self.font.render(text, True, (220, 220, 220)), (10, 45))
        pygame.display.flip()
    
def override_configs(env_cfg, args):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
    """
    task_name = args.task
    # override some parameters for testing
    # number of environments
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if "cts" in task_name:  # cts specific
        env_cfg.env.num_teacher = 1
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    # adjust parameters according to terrain type
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.env.debug_draw_terrain_height_points = False
        
        
        # random uniform terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.random_uniform_terrain", 
        #                                   "min_height" : -0.05, "max_height": 0.05, 
        #                                   "step":0.005, "downsampled_scale" : 0.2}
        # slope
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_sloped_terrain",
        #                                   "slope": -0.4, "platform_size": 3.0}
        # stairs
        env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_stairs_terrain",
                                        "step_width": 0.31, "step_height": -0.1, "platform_size": 3.0}
        # discrete obstacles
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.discrete_obstacles_terrain",
        #                                   "max_height": 0.1,
        #                                   "min_size": 1.0,
        #                                   "max_size": 2.0,
        #                                   "num_rects": 20,
        #                                   "platform_size": 3.0}
        # wave terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.wave_terrain", 
        #                                   "amplitude": 0.1, "num_waves": 2}
        # stepping stones
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.stepping_stones_terrain",
        #                                   "stone_size": 1.0, "max_height": 0.1,
        #                                   "stone_distance": 0.3, "platform_size": 3.0}
        # gap terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.gap_terrain", 
        #                                   "gap_size": 0.2, "platform_size": 3.0}
        # pit terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pit_terrain", 
        #                                   "depth": 0.2, "platform_size": 3.0}
        
        
    env_cfg.env.debug = True

    # For fixed-command debugging, disable heading mode so commands[:, 2]
    # remains the actual yaw-rate command seen by the policy/environment.
    if not args.use_joystick and not args.use_keyboard:
        env_cfg.commands.heading_command = False
    
    if args.use_joystick or args.use_keyboard:
        env_cfg.commands.heading_command = False

def print_debug_info(env, robot_index):
    """Print debug information while interacting

    Args:
        env: environment object
        robot_index (int): index of the robot to print info for
    """
    # print debug info
    # print("base lin vel: ", env.simulator.base_lin_vel[robot_index, :].cpu().numpy())
    # print("base yaw angle: ", env.simulator.base_euler[robot_index, 2].item())
    # print("base height: ", env.simulator.base_pos[robot_index, 2].cpu().numpy())
    # print("foot_height: ", env.simulator.feet_pos[robot_index, :, 2].cpu().numpy())
    # print(f"ankle pitch: {env.simulator.dof_pos[robot_index, [3,7]].cpu().numpy()}")
    # print(f"actions: {env.simulator.dof_pos[robot_index].cpu().numpy()}")
    # print(f"contact forces: {env.simulator.link_contact_forces[robot_index, env.simulator.feet_contact_indices, :].cpu().numpy()}")
    print(f"commands: {env.commands[robot_index].cpu().numpy()}")
    if hasattr(env, "stand_mask") and hasattr(env, "crawl_mask"):
        print(
            f"stand={int(env.stand_mask[robot_index].item())}, "
            f"crawl={int(env.crawl_mask[robot_index].item())}, "
            f"vx={env.simulator.base_lin_vel[robot_index, 0].item():.3f}"
        )
    pass

def interaction_loop(env, policy, args):
    """Run interaction loop between environment and policy

    Args:
        env: environment object
        policy : a policy that takes observations and outputs actions
        args: command line arguments
    """
    
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
        
    # Get initial observations according to task type
    task_name = args.task
    if "ts" in task_name or "cat" in task_name:  # teacher-student
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif "ee" in task_name:  # explicit estimator
        estimator_features, _, _ = env.get_observations()
    elif "dreamwaq" in task_name:  # dreamwaq
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    else: # vanilla
        obs = env.get_observations()
    
    # Setup joystick if needed
    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)
    keyboard = None
    if args.use_keyboard:
        if args.use_joystick:
            raise ValueError("Use either --use_joystick or --use_keyboard, not both")
        keyboard = KeyboardControl()
    
    frame_dt = 1 / 60.0 # 30Hz
    # interaction loop
    for i in range(10*int(env.max_episode_length)):
        
        t_start = time.perf_counter()
        # update commands from joystick
        if args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly
            env.commands[:, 1] = -joystick.lx
            env.commands[:, 2] = -joystick.rx
        elif args.use_keyboard:
            keyboard.update(env)
        else:
            env.commands[:, 0] = 0.1
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0   
            env.commands[:, 3] = 0.0
        
        # set the viewer camera to follow the first environment by default
        if args.follow_robot:
            pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.pos, dtype=np.float32)
            lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.lookat, dtype=np.float32)
            env.set_viewer_camera(pos, lookat)
        
        # Step the environment according to task type
        if "ts" in task_name or "cat" in task_name:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif "ee" in task_name:
            actions = policy(estimator_features.detach())
            estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        elif "waq" in task_name:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())
        else:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
        
        # print debug info
        print_debug_info(env, robot_index)
        
        # Update logger info
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.simulator.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.simulator.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.simulator.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.simulator.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.simulator.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.simulator.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.simulator.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.simulator.link_contact_forces[robot_index, 
                                                                          env.simulator.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
        # sleep for the remainder of the frame budget to match real-time playback
        elapsed = time.perf_counter() - t_start
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

def export_policy(alg_runner, path: str, args, env_cfg, train_cfg):
    """export the policy as jit script according to different task types

    Args:
        alg_runner: algorithm runner
        path (str): path to which the policy is exported
        args: command line arguments
        env_cfg: environment configuration
        train_cfg: training configuration
    """
    task_name = args.task
    if "ts" in task_name or "cat" in task_name:
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "ee" in task_name:
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "dreamwaq" in task_name:
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)
    

def play(args):
    """Main function to run the play script

    Args:
        args (_type_): command line arguments
    """
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning',
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs(env_cfg, args)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++ or python)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 
                            train_cfg.runner.load_run, 'exported')
    export_policy(ppo_runner, path, args, env_cfg, train_cfg)

    interaction_loop(env, policy, args)
    
    
if __name__ == '__main__':
    args = get_args()
    play(args)
