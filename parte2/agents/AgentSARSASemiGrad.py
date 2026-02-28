import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim: int, nA: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nA),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AgentSarsaSemiGrad:
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1.0,
        decay: bool = True,
        discount_factor: float = 0.99,
        alpha: float = 1e-3,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        hidden: int = 128,
        device: str | None = None,
        seed: int = 0,
    ):
        self.env = env
        self.nA = env.action_space.n
        obs_dim = int(np.prod(env.observation_space.shape))

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.qnet = QNet(obs_dim, self.nA, hidden=hidden).to(self.device)

        # “alpha” aquí es lr del optimizador
        self.optim = optim.Adam(self.qnet.parameters(), lr=float(alpha))
        self.loss_fn = nn.MSELoss()

        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.gamma = float(discount_factor)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Estadísticas (parecidas a lo tuyo)
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.step_count = 0
        self.t = 0
        self.list_stats_success = []
        self.episode_return = 0.0

        # Reproducibilidad
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(obs).to(self.device)

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> int:
        """ε-greedy (entrenamiento)."""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.nA))
        s = self._to_tensor(state)
        q = self.qnet(s)  # (1, nA)
        return int(torch.argmax(q, dim=1).item())

    @torch.no_grad()
    def get_greedy_action(self, state: np.ndarray) -> int:
        """Greedy puro (evaluación)."""
        s = self._to_tensor(state)
        q = self.qnet(s)
        return int(torch.argmax(q, dim=1).item())

    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        SARSA semi-gradiente:
          target = r + gamma * q(s', a'; w)   (si no terminal)
          w <- w + alpha * (target - q(s,a;w)) * grad_w q(s,a;w)

        En PyTorch:
          minimizamos (q(s,a;w) - target)^2
          con target DETACH para semi-gradiente.
        """
        self.step_count += 1
        self.episode_return += float(reward)

        done = bool(terminated or truncated)

        s = self._to_tensor(state)
        q_s = self.qnet(s)                      # (1, nA)
        q_sa = q_s[0, int(action)]              # escalar

        if not done:
            next_action = self.get_action(next_state)
            sp = self._to_tensor(next_state)

            # q(s',a';w) pero el target se trata como constante => detach()
            q_sp = self.qnet(sp)[0, int(next_action)].detach()
            target = torch.tensor(float(reward), device=self.device) + self.gamma * q_sp
        else:
            next_action = None
            target = torch.tensor(float(reward), device=self.device)

        loss = self.loss_fn(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # fin de episodio
        if done:
            self.episode_lengths.append(self.step_count)

            # En Acrobot, terminated=True suele significar que alcanzó el objetivo.
            self.list_stats_success.append(1 if (terminated and not truncated) else 0)

            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            if self.decay:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        return next_action

    def get_stats(self):
        return self.list_stats, self.episode_lengths, self.list_stats_success