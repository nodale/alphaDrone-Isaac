import gymnasium as gym

gym.register(
    id="run",
    entry_point="env.drone_env:DroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "env.drone_env:DroneEnvCfg",
        "skrl_cfg_entry_point": "agent:skrl_ppo_cfg.yaml",
    },
)

