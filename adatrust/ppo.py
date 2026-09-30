"""Compact PPO with factorized categorical heads (paper Sec. 4.3, Alg. 2)."""
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEADS = [3, 4, 4, 4]          # protocol | block | timeout | committee hint


class ActorCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh())
        self.pi_heads = nn.ModuleList([nn.Linear(256, h) for h in HEADS])
        self.v_head = nn.Linear(256, 1)

    def forward(self, x):
        z = self.trunk(x)
        return [h(z) for h in self.pi_heads], self.v_head(z).squeeze(-1)


class PPO:
    def __init__(self, obs_dim, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                 ent=0.01, epochs=10, batch=2048):
        self.net = ActorCritic(obs_dim).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip, self.ent = gamma, lam, clip, ent
        self.epochs, self.batch = epochs, batch

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, v = self.net(x)
        acts, logps = [], 0.0
        for lg in logits:
            d = torch.distributions.Categorical(logits=lg)
            a = d.sample()
            acts.append(int(a.item()))
            logps = logps + d.log_prob(a)
        return acts, float(logps.item()), float(v.item())

    def bc_update(self, obs_list, pid_list, epochs=4):
        """Supervised warmup on the protocol head only (E5 v6/v7): distill a
        privileged teacher (TM while it serves; PBFT while TM is down) before
        PPO fine-tuning. Breaks the softmax-saturation deadlock in which the
        protocol head collapses before the first escape lesson is
        experienced."""
        obs = torch.as_tensor(np.array(obs_list), dtype=torch.float32,
                              device=DEVICE)
        pid = torch.as_tensor(np.array(pid_list), dtype=torch.long,
                              device=DEVICE)
        for _ in range(epochs):
            logits, _ = self.net(obs)
            d = torch.distributions.Categorical(logits=logits[0])
            loss = -d.log_prob(pid).mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()

    def update(self, buf):
        obs = torch.as_tensor(np.array(buf["obs"]), dtype=torch.float32, device=DEVICE)
        acts = torch.as_tensor(np.array(buf["acts"]), dtype=torch.long, device=DEVICE)
        old_logp = torch.as_tensor(buf["logp"], dtype=torch.float32, device=DEVICE)
        adv = torch.as_tensor(buf["adv"], dtype=torch.float32, device=DEVICE)
        ret = torch.as_tensor(buf["ret"], dtype=torch.float32, device=DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(obs)
        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.batch):
                mb = idx[s:s + self.batch]
                logits, v = self.net(obs[mb])
                logp, ent = 0.0, 0.0
                for h, lg in enumerate(logits):
                    d = torch.distributions.Categorical(logits=lg)
                    logp = logp + d.log_prob(acts[mb, h])
                    ent = ent + d.entropy().mean()
                ratio = torch.exp(logp - old_logp[mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[mb]
                loss = (-torch.min(s1, s2).mean() + 0.5 * ((v - ret[mb]) ** 2).mean()
                        - self.ent * ent)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()


def compute_gae(rewards, values, dones, gamma, lam):
    adv, gae = np.zeros_like(rewards), 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nxt * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    ret = adv + np.array(values[:len(rewards)])
    return adv, ret
