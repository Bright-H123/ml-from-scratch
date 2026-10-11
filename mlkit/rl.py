"""Reinforcement learning (Course 3, Week 3).

Covers the Mars-rover MDP, the Bellman equation, value iteration, tabular Q-learning
and the DQN loop used in the lunar-lander lab (here with the NumPy network from
mlkit.neural_net instead of Keras).

The framing that makes RL click: supervised learning needs (x, y) pairs, and for
"how much thrust should the lander apply right now" nobody can label y. What we can
specify is a REWARD -- +100 for landing, -100 for crashing -- and let the algorithm
figure out which actions lead there.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MarsRover",
    "value_iteration",
    "policy_from_q",
    "q_learning",
    "epsilon_greedy",
    "ReplayBuffer",
    "DQNAgent",
]


class MarsRover:
    """The course's 6-state Mars rover.

    States 0..n-1 in a line. States 0 and n-1 are terminal with rewards 100 and 40;
    everything in between gives 0. Actions: 0 = left, 1 = right.

    The point of the example is that the optimal action depends on the discount: with
    gamma close to 1 the far-off 100 is worth walking for, and with a small gamma the
    nearby 40 wins. Impatience is a parameter.
    """

    def __init__(self, n_states: int = 6, left_reward: float = 100.0,
                 right_reward: float = 40.0, misstep_prob: float = 0.0) -> None:
        self.n_states = n_states
        self.n_actions = 2
        self.rewards = np.zeros(n_states)
        self.rewards[0] = left_reward
        self.rewards[-1] = right_reward
        self.terminal = np.zeros(n_states, dtype=bool)
        self.terminal[0] = self.terminal[-1] = True
        # A stochastic environment: the rover sometimes goes the wrong way.
        self.misstep_prob = misstep_prob

    def step(self, state: int, action: int, rng: np.random.Generator | None = None) -> tuple[int, float, bool]:
        """Return (next_state, reward, done). Reward is collected on ENTERING a state."""
        if self.terminal[state]:
            return state, 0.0, True
        if self.misstep_prob and rng is not None and rng.random() < self.misstep_prob:
            action = 1 - action
        nxt = state - 1 if action == 0 else state + 1
        nxt = int(np.clip(nxt, 0, self.n_states - 1))
        return nxt, float(self.rewards[nxt]), bool(self.terminal[nxt])

    def transitions(self, state: int, action: int) -> list[tuple[float, int]]:
        """(probability, next_state) pairs — used by the exact solver."""
        if self.terminal[state]:
            return [(1.0, state)]
        intended = int(np.clip(state - 1 if action == 0 else state + 1, 0, self.n_states - 1))
        if self.misstep_prob == 0:
            return [(1.0, intended)]
        other = int(np.clip(state + 1 if action == 0 else state - 1, 0, self.n_states - 1))
        return [(1.0 - self.misstep_prob, intended), (self.misstep_prob, other)]


def value_iteration(env: MarsRover, gamma: float = 0.9, tol: float = 1e-10,
                    max_iters: int = 10_000) -> np.ndarray:
    """Solve Q(s,a) exactly by repeatedly applying the Bellman equation.

        Q(s, a) = R(s) + gamma * max_{a'} Q(s', a')

    Read it as: the value of taking a in s is the reward you get right now, plus the
    discounted value of behaving optimally from wherever you land. The max over a' is
    the "and then act optimally" part -- it is what makes Q a statement about the best
    achievable return rather than about any particular policy.

    Iterating this contraction converges to the unique fixed point for gamma < 1.

    (Convention note: the course writes R(s) as the reward of the state you are IN.
    Here the reward comes from the state you ENTER, which is the same thing shifted by
    one step and keeps terminal rewards where you expect them.)
    """
    Q = np.zeros((env.n_states, env.n_actions))
    for _ in range(max_iters):
        Q_new = np.zeros_like(Q)
        for s in range(env.n_states):
            if env.terminal[s]:
                continue                      # terminal states have no future
            for a in range(env.n_actions):
                total = 0.0
                for prob, s_next in env.transitions(s, a):
                    r = env.rewards[s_next]
                    future = 0.0 if env.terminal[s_next] else gamma * Q[s_next].max()
                    total += prob * (r + future)
                Q_new[s, a] = total
        if np.max(np.abs(Q_new - Q)) < tol:
            Q = Q_new
            break
        Q = Q_new
    return Q


def policy_from_q(Q: np.ndarray) -> np.ndarray:
    """pi(s) = argmax_a Q(s, a). Once Q is known, the policy is free."""
    return np.argmax(np.asarray(Q), axis=1)


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Greedy with probability 1 - epsilon, uniformly random with probability epsilon.

    Why explore at all: if the network's initial random weights happen to say "thruster
    1 is always bad", a purely greedy agent will never fire it and so will never get
    evidence to the contrary. Epsilon forces the agent to gather that evidence.

    Standard schedule: start near 1.0 (almost all exploration) and decay towards ~0.01.
    """
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))
    return int(np.argmax(q_values))


def q_learning(
    env: MarsRover,
    episodes: int = 2000,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: int = 0,
) -> np.ndarray:
    """Tabular Q-learning: learn Q from experience, without knowing the transitions.

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
                                     \\_______ TD target _______/

    The bracket is the temporal-difference error: how wrong the current estimate was.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    eps = epsilon_start
    for _ in range(episodes):
        s = int(rng.integers(1, env.n_states - 1))     # start somewhere non-terminal
        for _ in range(100):
            a = epsilon_greedy(Q[s], eps, rng)
            s_next, r, done = env.step(s, a, rng)
            target = r if done else r + gamma * Q[s_next].max()
            Q[s, a] += alpha * (target - Q[s, a])
            if done:
                break
            s = s_next
        eps = max(epsilon_min, eps * epsilon_decay)
    return Q


class ReplayBuffer:
    """Fixed-size circular store of (s, a, r, s', done) tuples.

    Two problems it solves, both mentioned in the lunar-lander lab:
      * consecutive steps within an episode are highly correlated, and training on
        them in order makes the updates strongly biased; sampling a random minibatch
        from a large buffer decorrelates them.
      * each experience gets reused many times instead of being seen once and thrown
        away, which matters when collecting experience is the expensive part.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self.data: list[tuple] = []
        self._pos = 0

    def add(self, s, a, r, s_next, done) -> None:
        item = (np.asarray(s, dtype=float), int(a), float(r), np.asarray(s_next, dtype=float), bool(done))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self._pos] = item
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.data), size=min(batch_size, len(self.data)))
        batch = [self.data[i] for i in idx]
        S = np.stack([b[0] for b in batch])
        A = np.array([b[1] for b in batch], dtype=int)
        R = np.array([b[2] for b in batch], dtype=float)
        S2 = np.stack([b[3] for b in batch])
        D = np.array([b[4] for b in batch], dtype=bool)
        return S, A, R, S2, D

    def __len__(self) -> int:
        return len(self.data)


class DQNAgent:
    """Deep Q-Network: a neural network replaces the Q table for continuous states.

    Architecture follows the course's lunar lander: input = state, output = one Q value
    per action (rather than feeding (s, a) in and getting one number out). That layout
    gets max_a' Q(s', a') from a single forward pass instead of one pass per action.

    Training set construction:
        x = s
        y = r                                  if s' is terminal
          = r + gamma * max_a' Q_target(s', a') otherwise

    y is computed from a SEPARATE target network. Without it, every update moves the
    very target it is chasing, and training oscillates or diverges. The target network
    tracks the online one by a soft update, Q_target <- tau*Q + (1-tau)*Q_target with
    tau small, so the target drifts slowly enough to be a stable reference.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (64, 64),
        gamma: float = 0.99,
        alpha: float = 1e-3,
        tau: float = 0.01,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        seed: int = 0,
    ) -> None:
        from mlkit.neural_net import NeuralNetwork

        sizes = [state_dim, *hidden, n_actions]
        acts = ["relu"] * len(hidden) + ["linear"]
        self.q = NeuralNetwork(sizes, acts, seed=seed)
        self.q_target = NeuralNetwork(sizes, acts, seed=seed)
        self._sync_target(1.0)
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.n_actions = n_actions
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def _sync_target(self, tau: float) -> None:
        for layer, target_layer in zip(self.q.layers, self.q_target.layers):
            target_layer.W = tau * layer.W + (1 - tau) * target_layer.W
            target_layer.b = tau * layer.b + (1 - tau) * target_layer.b

    def act(self, state: np.ndarray, epsilon: float) -> int:
        q_values = self.q.forward(np.atleast_2d(state))[0]
        return epsilon_greedy(q_values, epsilon, self.rng)

    def learn(self) -> float | None:
        """One gradient step on a replayed minibatch. Returns the TD loss, or None."""
        if len(self.buffer) < self.batch_size:
            return None
        S, A, R, S2, D = self.buffer.sample(self.batch_size, self.rng)

        q_next = self.q_target.forward(S2).max(axis=1)
        targets = R + self.gamma * q_next * (~D)

        # Only the taken action's output should receive gradient. Building the full
        # target vector from the current predictions and overwriting one entry makes
        # the error on every other action exactly zero.
        q_pred = self.q.forward(S)
        y = q_pred.copy()
        y[np.arange(len(A)), A] = targets

        if self.q._t == 0:
            self.q._init_adam()
        self.q.backward(S, y)
        self.q._adam_step(self.alpha, 0.9, 0.999, 1e-8)
        self._sync_target(self.tau)
        return float(np.mean((q_pred[np.arange(len(A)), A] - targets) ** 2))
