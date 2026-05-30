from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

#base action class
class Action(ABC):

    def __init__(self, num_envs, dim, device):
        self.num_envs = num_envs
        self.dim = dim
        self.device = device

    @abstractmethod
    def reset(self, ids, start_pos, generator):
        pass

    @abstractmethod
    def step(self, ids, tau, out):
        pass

class HoldPosition(Action):

    def __init__(self, num_envs, dim, device):
        super().__init__(num_envs, dim, device)
        self.target = torch.zeros(num_envs, dim, device=device)

    def reset(self, ids, start_pos, generator):
        self.target[ids] = start_pos

    def step(self, ids, tau, out):
        out[ids] = self.target[ids]

class RandomWalk(Action):

    def __init__(self, num_envs, dim, device, vel_scale=0.5):
        super().__init__(num_envs, dim, device)

        self.vel_scale = vel_scale

        self.p0 = torch.zeros(num_envs, dim, device=device)
        self.vel = torch.zeros(num_envs, dim, device=device)

    def reset(self, ids, start_pos, generator):

        n = ids.numel()

        self.p0[ids] = start_pos

        self.vel[ids] = (
            torch.randn(
                n,
                self.dim,
                generator=generator,
                device=self.device,
            )
            * self.vel_scale
        )

    def step(self, ids, tau, out):

        t = tau[ids].unsqueeze(-1)

        out[ids] = (
            self.p0[ids]
            + self.vel[ids] * t
        )

class CubicSpline(Action):

    def __init__(self, num_envs, dim, device, scale=0.5):
        super().__init__(num_envs, dim, device)

        self.scale = scale

        self.p0 = torch.zeros(num_envs, dim, device=device)
        self.p1 = torch.zeros(num_envs, dim, device=device)
        self.p2 = torch.zeros(num_envs, dim, device=device)
        self.p3 = torch.zeros(num_envs, dim, device=device)

    def _noise(self, n, generator):

        return (
            torch.randn(
                n,
                self.dim,
                generator=generator,
                device=self.device,
            )
            * self.scale
        )

    def reset(self, ids, start_pos, generator):

        n = ids.numel()

        n1 = self._noise(n, generator)
        n2 = self._noise(n, generator)
        n3 = self._noise(n, generator)

        self.p0[ids] = start_pos
        self.p1[ids] = start_pos + n1
        self.p2[ids] = start_pos + n1 + n2
        self.p3[ids] = start_pos + n1 + n2 + n3

    def step(self, ids, tau, out):

        t = tau[ids].unsqueeze(-1)
        omt = 1.0 - t

        omt2 = omt * omt
        omt3 = omt2 * omt

        t2 = t * t
        t3 = t2 * t

        out[ids] = (
            omt3 * self.p0[ids]
            + 3.0 * omt2 * t * self.p1[ids]
            + 3.0 * omt * t2 * self.p2[ids]
            + t3 * self.p3[ids]
        )

class RandomSphereOffset(Action):

    def __init__(
        self,
        num_envs,
        dim,
        device,
        min_radius=0.1,
        max_radius=1.0,
    ):
        super().__init__(num_envs, dim, device)

        self.min_radius = min_radius
        self.max_radius = max_radius

        self.target = torch.zeros(num_envs, dim, device=device)

    def reset(self, ids, start_pos, generator):
        n = len(ids)

        direction = torch.randn(
            (n, self.dim),
            generator=generator,
            device=self.device,
        )

        direction = direction / torch.norm(
            direction,
            dim=-1,
            keepdim=True,
        )

        radius = torch.rand(
            (n, 1),
            generator=generator,
            device=self.device,
        )

        radius = (
            self.min_radius
            + (self.max_radius - self.min_radius) * radius
        )

        # Offset
        offset = direction * radius

        # Final target
        self.target[ids] = start_pos + offset

    def step(self, ids, tau, out):
        out[ids] = self.target[ids]

#action manager
@dataclass
class ActionPrimitive:
    actions: list[type[Action]]
    generator: torch.Generator
    dim: int
    num_envs: int
    device: str = "cuda"
    min_duration: int = 50
    max_duration: int = 200
    min_z: float = 0.6


    def __post_init__(self):
        self.bank = [a(dim=self.dim, device=self.device, num_envs=self.num_envs) for a in self.actions]

        self.current = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.t = torch.zeros(self.num_envs, device=self.device)
        self.duration = self._sample_duration()

        all_envs = torch.arange(
            self.num_envs,
            device=self.device,
        )

        self._switch(
            all_envs,
            current_pos=torch.zeros(
                self.num_envs,
                self.dim,
                device=self.device,
            ),
        )

    def _sample_duration(self):
        return torch.randint(
            self.min_duration,
            self.max_duration,
            (self.num_envs,),
            generator=self.generator,
            device=self.device,
        )

    def _switch(self, env_ids, current_pos):
        if len(env_ids) == 0:
            return

        next_actions = torch.randint(
            0,
            len(self.bank),
            (len(env_ids),),
            generator=self.generator,
            device=self.device,
        )

        self.current[env_ids] = next_actions
        self.t[env_ids] = 0.0
        self.duration[env_ids] = torch.randint(
            self.min_duration,
            self.max_duration,
            (env_ids.numel(),),
            generator=self.generator,
            device=self.device,
        )

        for idx, action in enumerate(self.bank):
            mask = next_actions == idx

            if mask.any():
                ids = env_ids[mask]

                action.reset(
                    ids=ids,
                    start_pos=current_pos[ids],
                    generator=self.generator,
                )

    @torch.no_grad()
    def step(self, current_pos):

        expired = torch.where(self.t >= self.duration)[0]
        self._switch(expired, current_pos)

        tau = self.t / self.duration.clamp_min(1)

        out = torch.empty_like(current_pos)

        for idx, action in enumerate(self.bank):

            ids = torch.nonzero(
                self.current == idx,
                as_tuple=False,
            ).squeeze(-1)

            if ids.numel() > 0:
                action.step(ids, tau, out)

        self.t += 1

        z = out[:, 2]
        below = z < self.min_z
        z[below] = self.min_z + (self.min_z - z[below])

        return out

#only for testing
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    primitive = ActionPrimitive(
        actions=[
            HoldPosition,
            RandomWalk,
            CubicSpline,
        ],
        dim=3,
        num_envs=4096,
        device=device,
        seed=42,
        min_duration=20,
        max_duration=100,
    )

    # external simulator state
    pos = torch.zeros(4096, 3, device=device)

    for _ in range(1000):

        # target setpoints
        target = primitive.step(pos)

        # fake dynamics
        pos += 0.02 * (target - pos)
