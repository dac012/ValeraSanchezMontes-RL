import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# ==========================================================
# 1. RED NEURONAL
# ==========================================================
class DQN_Network(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQN_Network, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, num_actions)
        )

        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        return self.FC(x)

# ==========================================================
# 2. AGENTE n-step Semi-Gradient SARSA (corregido para Acrobot)
#   - on-policy real: update devuelve next_action
#   - truncated cuenta como done para cortar el retorno
#   - flush interno al terminar episodio (sin flush dummy)
# ==========================================================
class AgentNStepSemiGradientSARSA:
    def __init__(self, env: gym.Env,
                 n: int = 3,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 discount_factor: float = 0.99,
                 lr: float = 3e-4,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995,
                 grad_clip_norm: float | None = 10.0):

        self.env = env
        self.nA = env.action_space.n
        self.nS = env.observation_space.shape[0]

        self.n = int(n)
        self.gamma = float(discount_factor)
        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.grad_clip_norm = grad_clip_norm

        self.q_network = DQN_Network(self.nA, self.nS)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, done)
        self.n_step_buffer = deque(maxlen=self.n)

        self.t = 0
        self.T = np.inf

        # Stats compatibles con tu estilo
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.list_stats_success = []
        self.step_count = 0
        self.episode_return = 0.0
        self.episode_idx = 0

    # ======================================================
    # ε-greedy
    # ======================================================
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def get_greedy_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    # ======================================================
    # UPDATE: n-step semi-gradient SARSA
    # Devuelve next_action para que el notebook la ejecute (on-policy real)
    # ======================================================
    def update(self, state, action, next_state, reward, terminated, truncated, info):
        done = terminated or truncated

        # inicio episodio
        if self.t == 0:
            self.T = np.inf
            self.n_step_buffer.clear()

        # seleccionar A_{t+1} una sola vez
        if done:
            self.T = self.t + 1
            next_action = None
        else:
            next_action = self.get_action(next_state)

        # Guardar transición usando done (incluye truncated)
        self.n_step_buffer.append((state, action, reward, next_state, next_action, done))

        self.step_count += 1
        self.episode_return += reward

        # τ ← t − n + 1
        tau = self.t - self.n + 1
        if tau >= 0:
            self._update_tau()

        self.t += 1

        # flush interno al terminar episodio
        if done:
            while len(self.n_step_buffer) > 0:
                self._update_tau()

            # stats
            self.episode_lengths.append(self.step_count)
            self.list_stats_success.append(1 if terminated and not truncated else 0)

            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.episode_idx + 1))

            # decay
            if self.decay:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # reset episodio
            self.t = 0
            self.step_count = 0
            self.episode_return = 0.0
            self.episode_idx += 1
            self.n_step_buffer.clear()

            return None

        return next_action

    # ======================================================
    def _update_tau(self):
        s_tau, a_tau, _, _, _, _ = self.n_step_buffer[0]

        G = 0.0
        gamma_pow = 1.0

        for i, (s, a, r, s_next, a_next, done_flag) in enumerate(self.n_step_buffer):
            G += gamma_pow * float(r)
            gamma_pow *= self.gamma

            if done_flag:
                break

            # bootstrap solo si tenemos n pasos completos y no terminal
            if i == self.n - 1 and a_next is not None:
                with torch.no_grad():
                    s_next_t = torch.FloatTensor(s_next).unsqueeze(0)
                    q_boot = self.q_network(s_next_t)[0, int(a_next)].item()
                G += gamma_pow * q_boot
                break

        s_tau_t = torch.FloatTensor(s_tau).unsqueeze(0)
        q_pred = self.q_network(s_tau_t)[0, int(a_tau)]
        target = torch.tensor(G, dtype=torch.float32)

        loss = self.loss_fn(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        self.n_step_buffer.popleft()

    def get_stats(self):
        return self.q_network, self.list_stats, self.episode_lengths, self.list_stats_success