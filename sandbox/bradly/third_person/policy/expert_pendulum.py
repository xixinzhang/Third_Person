import sys
import os
sys.path.append('/home/asus/Workspace/Third_Person/')
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


from sandbox.bradly.third_person.envs.gym_env import GymEnv
from sandbox.bradly.third_person.envs.pendulum import PendulumEnv


os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_options = tf.GPUOptions(allow_growth=True)

def generate_expert_dp():
    env = TfEnv(normalize(GymEnv(PendulumEnv,{'color':'red'})))
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
        n_itr=640,
        discount=0.995,
        step_size=0.01,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        gae_lambda=0.97,

    )

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        algo.train(sess=sess)
        t = rollout(env=env, agent=policy, max_path_length=100, animated=False)
        print(sum(t['rewards']))
        with open('expert_pendulum.pickle', 'wb') as handle:
            pickle.dump(policy, handle)
        # while True:
        #     rollout(env=env, agent=policy, max_path_length=100, animated=True)

def load_expert_inverted_dp(f='expert_dp.pickle'):
    with open(f, 'rb') as handle:
        policy = pickle.load(handle)

    return policy

def test(f='expert_pendulum.pickle'):
    env = TfEnv(normalize(GymEnv(PendulumEnv,{'color':'red'})))
    policy=load_expert_inverted_dp(f)
    while True:
        rollout(env=env, agent=policy, max_path_length=100, animated=True)



if __name__ == '__main__':
    generate_expert_dp()
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     test()
    #with tf.Session() as sess:
    #    env = TfEnv(normalize(InvertedPendulumEnv()))
    #    expert = load_expert_reacher(env, sess)
    #    while True:
    #        t = rollout(env=env, agent=expert, max_path_length=100, animated=True)
