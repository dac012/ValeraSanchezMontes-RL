import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

class DQN_Network(nn.Module):
    def __init__(self, num_actions: int, input_dim: int):
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
        )
        for module in self.FC:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.FC(x)


class AgentSarsaSemiGrad:
    def __init__(self, env: gym.Env,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 discount_factor: float = 0.99,
                 alpha: float = 1e-3,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995,
                 success_threshold: float = 100.0,
                 normalize_obs: bool = False,
                 device: str | None = None):

        self.env = env
        self.nA = env.action_space.n

        assert len(env.observation_space.shape) == 1
        self.input_dim = env.observation_space.shape[0]  # LunarLander: 8

        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.gamma = float(discount_factor)
        self.alpha = float(alpha)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.success_threshold = float(success_threshold)
        self.normalize_obs = bool(normalize_obs)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.qnet = DQN_Network(num_actions=self.nA, input_dim=self.input_dim).to(self.device)
        self.optim = optim.Adam(self.qnet.parameters(), lr=self.alpha)

        # Stats
        self.episode_return = 0.0
        self.episode_length = 0
        self.returns = []
        self.lengths = []
        self.successes = []

    def _prep_obs(self, obs: np.ndarray) -> np.ndarray:
        # Normalización muy simple (opcional)
        if not self.normalize_obs:
            return obs
        # LunarLander obs: [x, y, vx, vy, angle, ang_vel, leg1, leg2]
        # Escalamos un poco velocidades y ángulo para evitar magnitudes distintas
        o = obs.copy().astype(np.float32)
        o[2] *= 0.1  # vx
        o[3] *= 0.1  # vy
        o[4] *= 1.0  # angle (ya suele ser pequeño)
        o[5] *= 0.1  # angular vel
        return o

    def _to_tensor(self, obs):
        obs = self._prep_obs(obs)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return int(self.env.action_space.sample())
        with torch.no_grad():
            qvals = self.qnet(self._to_tensor(state))
            return int(torch.argmax(qvals, dim=1).item())

    def get_greedy_action(self, state):
        with torch.no_grad():
            qvals = self.qnet(self._to_tensor(state))
            return int(torch.argmax(qvals, dim=1).item())

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        done = terminated or truncated

        self.episode_return += reward
        self.episode_length += 1

        # A' on-policy
        if not done:
            next_action = self.get_action(next_obs)
        else:
            next_action = None

        s = self._to_tensor(obs)
        s2 = self._to_tensor(next_obs)

        q_sa = self.qnet(s)[0, action]

        # target detached => semi-gradiente
        with torch.no_grad():
            r_t = torch.tensor(float(reward), device=self.device)
            if done:
                target = r_t
            else:
                q_s2a2 = self.qnet(s2)[0, next_action]
                target = r_t + self.gamma * q_s2a2

        loss = 0.5 * (target - q_sa) ** 2

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Fin episodio: guardar métricas y decaer epsilon
        if done:
            self.returns.append(self.episode_return)
            self.lengths.append(self.episode_length)
            self.successes.append(1 if self.episode_return >= self.success_threshold else 0)

            if self.decay:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.episode_return = 0.0
            self.episode_length = 0

        return next_action

    def stats(self):
        return {
            "returns": self.returns,
            "lengths": self.lengths,
            "successes": self.successes
        }