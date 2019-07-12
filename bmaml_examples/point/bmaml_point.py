import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__)))))))
from sandbox.rocky.tf.algos.bmaml_trpo import BMAMLTRPO 
from sandbox.rocky.tf.algos.bmaml_reptile import BMAMLREPTILE
from sandbox.rocky.tf.algos.bmaml_chaser import BMAMLCHASER
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from bmaml_examples.point.point_env_randgoal import PointEnvRandGoal
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.bmaml_minimal_gauss_mlp_policy import BMAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

# GENERAL
flags.DEFINE_integer('num_particles', 2, 'number of particles per task')
flags.DEFINE_integer('num_parallel', 8, 'number of parallel for sampler')
flags.DEFINE_integer('random_seed', 1, 'random seed')
flags.DEFINE_string('fast_lr', '0.1', 'few shot learning rate')
flags.DEFINE_string('meta_step_size', '0.01', 'meta learning rate')
flags.DEFINE_string('fast_batch_size', '10', 'few shot batch size')
flags.DEFINE_string('meta_batch_size', '20', 'meta batch size')
flags.DEFINE_integer('meta_iter', 100, 'meta_iter')
flags.DEFINE_string('meta_method', 'trpo', '{trpo|chaser|reptile}')
flags.DEFINE_string('mode', 'local', 'if you use aws, then you can choose other options (e.g. ec2)')
# SVPG
flags.DEFINE_string('method', 'svpg', '{svpg|vpg}')
flags.DEFINE_float('svpg_alpha', 1.0, 'svpg alpha')
# LOAD PREVIOUS WORK
flags.DEFINE_bool('load_policy', False, 'load previous works or not')

num_particles = FLAGS.num_particles
num_parallel = FLAGS.num_parallel
svpg = True if FLAGS.method=='svpg' else False
random_seed = FLAGS.random_seed
svpg_alpha = FLAGS.svpg_alpha
fast_learning_rate = float(FLAGS.fast_lr)
meta_step_size = float(FLAGS.meta_step_size)
fast_batch_size = int(FLAGS.fast_batch_size)
meta_batch_size = int(FLAGS.meta_batch_size)
meta_iter = FLAGS.meta_iter
meta_method = FLAGS.meta_method
mode = FLAGS.mode
load_policy = FLAGS.load_policy

# options
bas = 'linear'
max_path_length = 100
num_grad_updates = 1
num_leader_grad_updates = 2

# svpg str
if svpg:
    svpg_str = '_SVPG' + '_alpha' + str(svpg_alpha)
else:
    svpg_str = '_VPG'

# bmaml|emaml
if svpg == False:
    maml_type = 'emaml'
else:
    maml_type = 'bmaml'

exp_prefix_str = maml_type+'-'+meta_method+'-point100'
exp_name_str = maml_type+'_M'+str(num_particles)+svpg_str+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + '_mlr_' + str(meta_step_size) + '_randomseed'+str(random_seed)

stub(globals())

if load_policy:
    previous_work_path = os.path.join('data',mode,exp_prefix_str,exp_name_str)
    load_policy = previous_work_path+'/params_'
else:
    load_policy = None

env = TfEnv(normalize(PointEnvRandGoal()))    # env

if load_policy is None:
    policy_list = [BMAMLGaussianMLPPolicy(    # policy NN list
        name="policy",
        env_spec=env.spec,
        grad_step_size=fast_learning_rate,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
        particle_idx=i)
        for i in range(num_particles)
    ]
else:
    policy_list = [0]*num_particles

# baselines
if bas == 'zero':
    baseline_list = [ZeroBaseline(env_spec=env.spec) for i in range(num_particles)]
elif 'linear' in bas:
    baseline_list = [LinearFeatureBaseline(env_spec=env.spec) for i in range(num_particles)]
else:
    baseline_list = [GaussianMLPBaseline(env_spec=env.spec) for i in range(num_particles)]

if meta_method == 'trpo':
    algo = BMAMLTRPO(
        env=env,
        policy_list=policy_list,
        baseline_list=baseline_list,
        batch_size=fast_batch_size, # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=meta_batch_size,
        num_grad_updates=num_grad_updates,
        random_seed=random_seed,
        svpg=svpg,
        svpg_alpha=svpg_alpha,
        n_itr=meta_iter,
        step_size=meta_step_size,
        plot=False,
        load_policy=load_policy,
    )
elif meta_method == 'chaser':
    algo = BMAMLCHASER(
        env=env,
        policy_list=policy_list,
        baseline_list=baseline_list,
        batch_size=fast_batch_size, # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=meta_batch_size,
        num_grad_updates=num_grad_updates,
        num_leader_grad_updates=num_leader_grad_updates,
        random_seed=random_seed,
        svpg=svpg,
        svpg_alpha=svpg_alpha,
        n_itr=meta_iter,
        step_size=meta_step_size,
        plot=False,
        load_policy=load_policy,
    )
elif meta_method == 'reptile':
    algo = BMAMLREPTILE(
        env=env,
        policy_list=policy_list,
        baseline_list=baseline_list,
        batch_size=fast_batch_size, # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=meta_batch_size,
        num_grad_updates=num_grad_updates,
        num_leader_grad_updates=num_leader_grad_updates,
        random_seed=random_seed,
        svpg=svpg,
        svpg_alpha=svpg_alpha,
        n_itr=meta_iter,
        step_size=meta_step_size,
        plot=False,
        load_policy=load_policy,
    )
else:
    raise ValueError  # {trpo|chaser|chaser_trpo}

run_experiment_lite(
    algo.train(),
    n_parallel=num_parallel,
    snapshot_mode="last",
    python_command='python3',
    seed=1,
    exp_prefix=exp_prefix_str,
    exp_name=exp_name_str,
    plot=False,
)
