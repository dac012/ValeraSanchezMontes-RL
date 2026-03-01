import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim: int, nA: int, hidden: int = 12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nA),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AgentSarsaSemiGrad:
    def __init__(self,
                 env: gym.Env,
                 n: int = 1,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 decay_c: float = 1000.0,
                 discount_factor: float = 0.99,
                 alpha: float = 1e-3,
                 hidden: int = 12,
                 seed: int = 0,
                 device: str | None = None):

        self.env = env
        self.nA = env.action_space.n
        self.obs_dim = int(np.prod(env.observation_space.shape))

        self.n = int(n)
        if self.n < 1:
            raise ValueError("n debe ser >= 1")

        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.decay_c = float(decay_c)
        self.gamma = float(discount_factor)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.qnet = QNet(self.obs_dim, self.nA, hidden=hidden).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=float(alpha))
        self.loss_fn = nn.MSELoss()

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.S = [None] * (self.n + 1)
        self.A = [None] * (self.n + 1)
        self.R = [0.0] * (self.n + 1)

        self.T = None

        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.step_count = 0
        self.t = 0
        self.list_stats_success = []
        self.episode_return = 0.0

        self.last_terminated = False
        self.last_truncated = False


    def _to_tensor(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(obs).to(self.device)


    # Select and store an action A_t ~ ε-greedy wrt q̂(S_t, ·; w)
    @torch.no_grad()
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.nA))

        s = self._to_tensor(state)
        q_values = self.qnet(s)
        return int(torch.argmax(q_values, dim=1).item())


    @torch.no_grad()
    def get_greedy_action(self, state):
        s = self._to_tensor(state)
        q_values = self.qnet(s)
        return int(torch.argmax(q_values, dim=1).item())


    # Initialize and store S0 != terminal
    # Select and store A0
    # T <- ∞
    def start_episode(self, s0):
        self.S[0] = s0
        self.A[0] = self.get_action(s0)
        self.T = np.inf
        self.step_count = 0
        self.episode_return = 0.0


    # Loop for t = 0,1,2,...
    def step(self, t):
        self.step_count += 1

        # If t < T then:
        #   Take action A_t
        #   Observe and store R_{t+1} and S_{t+1}
        if t < self.T:

            s_t = self.S[t % (self.n + 1)]
            a_t = int(self.A[t % (self.n + 1)])

            s_tp1, r_tp1, terminated, truncated, info = self.env.step(a_t)

            self.last_terminated = bool(terminated)
            self.last_truncated = bool(truncated)

            self.R[(t + 1) % (self.n + 1)] = float(r_tp1)
            self.S[(t + 1) % (self.n + 1)] = s_tp1

            self.episode_return += float(r_tp1)

            done = bool(terminated or truncated)

            # If S_{t+1} is terminal then T <- t+1
            if done:
                self.T = t + 1
            else:
                # else Select and store A_{t+1}
                self.A[(t + 1) % (self.n + 1)] = self.get_action(s_tp1)

        # τ <- t - n + 1
        tau = t - self.n + 1

        # If τ >= 0 then:
        if tau >= 0:

            # G <- sum_{i=τ+1}^{min(τ+n, T)} γ^{i-τ-1} R_i
            G = 0.0
            upper = int(min(tau + self.n, self.T))
            for i in range(tau + 1, upper + 1):
                G += (self.gamma ** (i - tau - 1)) * self.R[i % (self.n + 1)]

            # If τ + n < T then:
            #   G <- G + γ^n q̂(S_{τ+n}, A_{τ+n}, w)
            if (tau + self.n) < self.T:
                s_boot = self.S[(tau + self.n) % (self.n + 1)]
                a_boot = int(self.A[(tau + self.n) % (self.n + 1)])

                with torch.no_grad():
                    q_boot = self.qnet(self._to_tensor(s_boot))[0, a_boot].item()

                G += (self.gamma ** self.n) * q_boot

            # w <- w + α [G - q̂(S_τ, A_τ, w)] ∇ q̂(S_τ, A_τ, w)
            s_tau = self.S[tau % (self.n + 1)]
            a_tau = int(self.A[tau % (self.n + 1)])

            q_values = self.qnet(self._to_tensor(s_tau))
            q_sa = q_values[0, a_tau]

            target = torch.tensor(G, dtype=torch.float32, device=self.device)

            loss = self.loss_fn(q_sa, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Until τ = T - 1
        done_episode = (tau == (self.T - 1))

        if done_episode:

            self.episode_lengths.append(self.step_count)

            if self.last_terminated and not self.last_truncated:
                self.list_stats_success.append(1)
            else:
                self.list_stats_success.append(0)

            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            if self.decay:
                self.epsilon = min(1.0, self.decay_c / (self.t + 1))

            self.t += 1

        return done_episode


    def get_stats(self):
        return self.list_stats, self.episode_lengths, self.list_stats_success