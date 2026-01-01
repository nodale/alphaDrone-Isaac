import gymnasium as gym

gym.register(
    id="run",
    entry_point=f"baseRunner:DroneRunner",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"baseEnvironment:DroneEnvironment",
    },
)

