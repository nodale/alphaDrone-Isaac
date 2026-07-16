from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import torch


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

    def __init__(self, num_envs, dim, device, vel_scale=0.4):
        super().__init__(num_envs, dim, device)
        self.vel_scale = vel_scale
        self.p0 = torch.zeros(num_envs, dim, device=device)
        self.vel = torch.zeros(num_envs, dim, device=device)

    def reset(self, ids, start_pos, generator):
        n = ids.numel()
        self.p0[ids] = start_pos
        self.vel[ids] = torch.randn(n, self.dim, generator=generator, device=self.device) * self.vel_scale

    def step(self, ids, tau, out):
        t = tau[ids].unsqueeze(-1)
        out[ids] = self.p0[ids] + self.vel[ids] * t


class CubicSpline(Action):

    def __init__(self, num_envs, dim, device, scale=0.5):
        super().__init__(num_envs, dim, device)
        self.scale = scale
        self.p0 = torch.zeros(num_envs, dim, device=device)
        self.p1 = torch.zeros(num_envs, dim, device=device)
        self.p2 = torch.zeros(num_envs, dim, device=device)
        self.p3 = torch.zeros(num_envs, dim, device=device)

    def _noise(self, n, generator):
        return torch.randn(n, self.dim, generator=generator, device=self.device) * self.scale

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
        out[ids] = (
            omt ** 3 * self.p0[ids]
            + 3.0 * omt ** 2 * t * self.p1[ids]
            + 3.0 * omt * t ** 2 * self.p2[ids]
            + t ** 3 * self.p3[ids]
        )


class RandomSphereOffset(Action):

    def __init__(self, num_envs, dim, device, min_radius=0.05, max_radius=0.4):
        super().__init__(num_envs, dim, device)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.target = torch.zeros(num_envs, dim, device=device)

    def reset(self, ids, start_pos, generator):
        n = len(ids)
        direction = torch.randn((n, self.dim), generator=generator, device=self.device)
        direction = direction / torch.norm(direction, dim=-1, keepdim=True)
        radius = torch.rand((n, 1), generator=generator, device=self.device)
        radius = self.min_radius + (self.max_radius - self.min_radius) * radius
        self.target[ids] = start_pos + direction * radius

    def step(self, ids, tau, out):
        out[ids] = self.target[ids]


class CircularOrbit(Action):
    """Horizontal circle centred at start_pos. Produces sustained centripetal acceleration."""

    def __init__(self, num_envs, dim, device, min_radius=0.05, max_radius=0.3,
                 min_laps=0.05, max_laps=0.5):
        super().__init__(num_envs, dim, device)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_laps = min_laps
        self.max_laps = max_laps
        self.center = torch.zeros(num_envs, dim, device=device)
        self.radius = torch.zeros(num_envs, device=device)
        self.total_angle = torch.zeros(num_envs, device=device)

    def reset(self, ids, start_pos, generator):
        n = ids.numel()
        r = self.min_radius + (self.max_radius - self.min_radius) * \
            torch.rand(n, generator=generator, device=self.device)
        laps = self.min_laps + (self.max_laps - self.min_laps) * \
            torch.rand(n, generator=generator, device=self.device)
        sign = (torch.rand(n, generator=generator, device=self.device) > 0.5).float() * 2 - 1
        self.center[ids] = start_pos
        self.radius[ids] = r
        self.total_angle[ids] = laps * 2 * math.pi * sign

    def step(self, ids, tau, out):
        t = tau[ids]
        phi = self.total_angle[ids] * t
        r = self.radius[ids]
        out[ids, 0] = self.center[ids, 0] + r * torch.cos(phi)
        out[ids, 1] = self.center[ids, 1] + r * torch.sin(phi)
        out[ids, 2] = self.center[ids, 2]


class HelixClimb(Action):
    """Ascending/descending spiral: circular XY motion + linear Z ramp."""

    def __init__(self, num_envs, dim, device, min_radius=0.05, max_radius=0.3,
                 min_laps=0.2, max_laps=1.0, min_dz=0.02, max_dz=0.5):
        super().__init__(num_envs, dim, device)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_laps = min_laps
        self.max_laps = max_laps
        self.min_dz = min_dz
        self.max_dz = max_dz
        self.center = torch.zeros(num_envs, dim, device=device)
        self.radius = torch.zeros(num_envs, device=device)
        self.total_angle = torch.zeros(num_envs, device=device)
        self.dz = torch.zeros(num_envs, device=device)

    def reset(self, ids, start_pos, generator):
        n = ids.numel()
        r = self.min_radius + (self.max_radius - self.min_radius) * \
            torch.rand(n, generator=generator, device=self.device)
        laps = self.min_laps + (self.max_laps - self.min_laps) * \
            torch.rand(n, generator=generator, device=self.device)
        sign = (torch.rand(n, generator=generator, device=self.device) > 0.5).float() * 2 - 1
        dz = self.min_dz + (self.max_dz - self.min_dz) * \
            torch.rand(n, generator=generator, device=self.device)
        dz_sign = (torch.rand(n, generator=generator, device=self.device) > 0.5).float() * 2 - 1
        self.center[ids] = start_pos
        self.radius[ids] = r
        self.total_angle[ids] = laps * 2 * math.pi * sign
        self.dz[ids] = dz * dz_sign

    def step(self, ids, tau, out):
        t = tau[ids]
        phi = self.total_angle[ids] * t
        r = self.radius[ids]
        out[ids, 0] = self.center[ids, 0] + r * torch.cos(phi)
        out[ids, 1] = self.center[ids, 1] + r * torch.sin(phi)
        out[ids, 2] = self.center[ids, 2] + self.dz[ids] * t


class LissajousPath(Action):
    """3D Lissajous figure with incommensurate frequency ratios across all three axes."""

    _FREQ_RATIOS = ((1, 2, 3), (2, 3, 5), (3, 4, 7), (1, 3, 5))

    def __init__(self, num_envs, dim, device, amplitude=0.6):
        super().__init__(num_envs, dim, device)
        self.amplitude = amplitude
        self.center = torch.zeros(num_envs, dim, device=device)
        self.amp = torch.zeros(num_envs, dim, device=device)
        self.freq = torch.zeros(num_envs, dim, device=device)
        self.phase = torch.zeros(num_envs, dim, device=device)
        self._freq_table = torch.tensor(self._FREQ_RATIOS, dtype=torch.float32, device=device)

    def reset(self, ids, start_pos, generator):
        n = ids.numel()
        self.center[ids] = start_pos
        self.amp[ids] = self.amplitude * (0.5 + 0.5 * torch.rand(
            n, self.dim, generator=generator, device=self.device))
        ratio_idx = torch.randint(0, len(self._FREQ_RATIOS), (n,),
                                  generator=generator, device=self.device)
        self.freq[ids] = self._freq_table[ratio_idx]
        self.phase[ids] = torch.rand(n, self.dim, generator=generator,
                                     device=self.device) * 2 * math.pi

    def step(self, ids, tau, out):
        t = tau[ids].unsqueeze(-1)
        out[ids] = self.center[ids] + self.amp[ids] * torch.sin(
            2 * math.pi * self.freq[ids] * t + self.phase[ids])


class ZigZag(Action):
    """Bounces between two random waypoints using a triangle wave, producing sharp reversals."""

    def __init__(self, num_envs, dim, device, segment_length=0.2, num_bounces=2):
        super().__init__(num_envs, dim, device)
        self.segment_length = segment_length
        self.num_bounces = num_bounces
        self.p0 = torch.zeros(num_envs, dim, device=device)
        self.p1 = torch.zeros(num_envs, dim, device=device)

    def reset(self, ids, start_pos, generator):
        n = ids.numel()
        direction = torch.randn(n, self.dim, generator=generator, device=self.device)
        direction[:, 2] *= 0.3
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        self.p0[ids] = start_pos
        self.p1[ids] = start_pos + direction * self.segment_length

    def step(self, ids, tau, out):
        t_scaled = torch.fmod(tau[ids] * self.num_bounces, 1.0)
        tri = (1 - torch.abs(2 * t_scaled - 1)).unsqueeze(-1)
        out[ids] = self.p0[ids] + tri * (self.p1[ids] - self.p0[ids])


class SinusoidalWalk(Action):
    """High-velocity drift with a sinusoidal acceleration ripple layered on top."""

    def __init__(self, num_envs, dim, device, vel_scale=0.5,
                 modulation_freq=1.5, modulation_depth=0.4):
        super().__init__(num_envs, dim, device)
        self.vel_scale = vel_scale
        self.modulation_freq = modulation_freq
        self.modulation_depth = modulation_depth
        self.p0 = torch.zeros(num_envs, dim, device=device)
        self.vel = torch.zeros(num_envs, dim, device=device)

    def reset(self, ids, start_pos, generator):
        n = ids.numel()
        self.p0[ids] = start_pos
        self.vel[ids] = torch.randn(n, self.dim, generator=generator,
                                    device=self.device) * self.vel_scale

    def step(self, ids, tau, out):
        t = tau[ids].unsqueeze(-1)
        f = self.modulation_freq
        d = self.modulation_depth
        mod = t + (d / (2 * math.pi * f)) * (1 - torch.cos(2 * math.pi * f * t))
        out[ids] = self.p0[ids] + self.vel[ids] * mod


@dataclass
class ActionPrimitive:
    actions: list
    generator: torch.Generator
    dim: int
    num_envs: int
    device: str = "cuda"
    min_duration: int = 50
    max_duration: int = 200
    min_xyz: tuple = (-2.5, -2.5, 0.5)
    max_xyz: tuple = (2.5, 2.5, 2.5)
    recenter: float = 0.15  # partial pull of the re-anchor point toward box center per switch

    def __post_init__(self):
        self.bank = [a(dim=self.dim, device=self.device, num_envs=self.num_envs) for a in self.actions]

        self.current = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.t = torch.zeros(self.num_envs, device=self.device)
        self.duration = self._sample_duration()

        self.min_xyz = torch.tensor(self.min_xyz, device=self.device, dtype=torch.float32)
        self.max_xyz = torch.tensor(self.max_xyz, device=self.device, dtype=torch.float32)
        self.center = (self.min_xyz + self.max_xyz) / 2

        all_envs = torch.arange(self.num_envs, device=self.device)
        self._switch(all_envs, current_pos=torch.zeros(self.num_envs, self.dim, device=self.device))

    def _sample_duration(self):
        return torch.randint(
            self.min_duration, self.max_duration,
            (self.num_envs,), generator=self.generator, device=self.device,
        )

    def _switch(self, env_ids, current_pos):
        if len(env_ids) == 0:
            return
        next_actions = torch.randint(
            0, len(self.bank), (len(env_ids),),
            generator=self.generator, device=self.device,
        )
        self.current[env_ids] = next_actions
        self.t[env_ids] = 0.0
        self.duration[env_ids] = torch.randint(
            self.min_duration, self.max_duration,
            (env_ids.numel(),), generator=self.generator, device=self.device,
        )

        # Bound the re-anchor point to the box, then pull it gently toward center so
        # segment-to-segment excursions can't accumulate into unbounded drift. This is
        # a partial blend (not a reset): each primitive still samples its own random
        # motion around this anchor, so trajectories stay varied.
        anchor = torch.max(torch.min(current_pos, self.max_xyz), self.min_xyz)
        anchor = (1.0 - self.recenter) * anchor + self.recenter * self.center

        for idx, action in enumerate(self.bank):
            mask = next_actions == idx
            if mask.any():
                ids = env_ids[mask]
                action.reset(ids=ids, start_pos=anchor[ids], generator=self.generator)

    @torch.no_grad()
    def step(self, current_pos, active_mask=None):
        if active_mask is None:
            active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.t[active_mask] += 1
        expired = torch.where((self.t >= self.duration) & active_mask)[0]
        self._switch(expired, current_pos)

        tau = self.t / self.duration.clamp_min(1)
        out = torch.empty_like(current_pos)

        for idx, action in enumerate(self.bank):
            ids = torch.nonzero((self.current == idx) & active_mask, as_tuple=False).squeeze(-1)
            if ids.numel() > 0:
                action.step(ids, tau, out)

        inactive = ~active_mask
        out[inactive] = current_pos[inactive]

        below = out < self.min_xyz
        above = out > self.max_xyz
        out[below] = 2 * self.min_xyz.expand_as(out)[below] - out[below]
        out[above] = 2 * self.max_xyz.expand_as(out)[above] - out[above]
        # Hard clamp guarantees in-box even for excursions wider than the box, where a
        # single reflection would otherwise land past the opposite wall.
        out = torch.max(torch.min(out, self.max_xyz), self.min_xyz)

        return out

    @torch.no_grad()
    def reset(self, env_ids, current_pos):
        self._switch(env_ids, current_pos)
