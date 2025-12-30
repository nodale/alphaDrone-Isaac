import gymnasium as gym

gym.register(
    id="run",
    entry_point=f"alphaDroneEnv:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"alphaDroneEnv:QuadcopterEnvCfg",
        "skrl_cfg_entry_point": f"agent:skrl_ppo_cfg.yaml",
    },
)

