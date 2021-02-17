import sys
sys.path.append('//home/asus/Workspace/Third_Person')
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.bradly.third_person.envs.inverted_dp import InvertedPendulumEnv
from rllab.sampler.utils import rollout
import pickle
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_options = tf.GPUOptions(allow_growth=True)

def generate_expert_dp():
    env = TfEnv(normalize(InvertedPendulumEnv()))
    policy = GaussianMLPPolicy(
        name="expert_policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64, 64),
        std_hidden_sizes=(64, 64),
        adaptive_std=True,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=64,
        discount=0.995,
        step_size=0.01,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        gae_lambda=0.97,

    )

    with tf.Session() as sess:
        algo.train(sess=sess)
        t = rollout(env=env, agent=policy, max_path_length=100, animated=False)
        print(sum(t['rewards']))
        with open('expert_dp.pickle', 'wb') as handle:
            pickle.dump(policy, handle)
        while True:
            rollout(env=env, agent=policy, max_path_length=100, animated=False)

def load_expert_inverted_dp(f='expert_dp.pickle'):
    with open(f, 'rb') as handle:
        policy = pickle.load(handle)

    return policy

def test_expert_reacher(file='expert_reacher.pickle'):
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        env = TfEnv(normalize(InvertedPendulumEnv(size=500)))
        expert = load_expert_inverted_dp(file)
        for i in range(10):
            t = rollout(env=env, agent=expert, max_path_length=50, animated=True)
            print(np.mean(sum(t['rewards'])))


if __name__ == '__main__':
    # generate_expert_dp()
    test_expert_reacher('expert_dp.pickle')
    #with tf.Session() as sess:
    #    env = TfEnv(normalize(InvertedPendulumEnv()))
    #    expert = load_expert_reacher(env, sess)
    #    while True:
    #        t = rollout(env=env, agent=expert, max_path_length=100, animated=True)
