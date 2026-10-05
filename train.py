"""训练：PPO vs 立回机器人。运行：python train.py [总步数]"""

import sys
from stable_baselines3 import PPO
from fighting_env import FightingEnv


def evaluate(model, n=50):
    env = FightingEnv()
    w = l = d = 0
    for ep in range(n):
        obs, _ = env.reset(seed=90000 + ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a))
            done = term or trunc
        if env.p2.hp <= 0 < env.p1.hp:
            w += 1
        elif env.p1.hp <= 0 < env.p2.hp:
            l += 1
        else:
            d += 1
    return w, l, d


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
    env = FightingEnv()
    model = PPO("MlpPolicy", env, verbose=1,
                n_steps=2048, batch_size=256,
                learning_rate=3e-4, gamma=0.99,
                ent_coef=0.01, seed=42)
    model.learn(total_timesteps=steps)
    model.save("fighting_ppo")

    w, l, d = evaluate(model)
    print(f"\n评估 50 场 vs 立回机器人：胜 {w} / 负 {l} / 判 {d}  "
          f"（胜率含判定 {(w + 0.5 * d) / 50:.0%}）")


if __name__ == "__main__":
    main()
