import sys, os
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-3]))
from sandbox.rocky.tf.algos.bmaml_reptile import BMAMLREPTILE
from sandbox.rocky.tf.algos.bmaml_chaser import BMAMLCHASER
from sandbox.rocky.tf.algos.bmaml_trpo import BMAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.ant_env_rand import AntEnvRand
from rllab.envs.mujoco.ant_env_rand_goal import AntEnvRandGoal
from rllab.envs.mujoco.ant_env_rand_direc import AntEnvRandDirec
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
flags.DEFINE_string('fast_batch_size', '20', 'few shot batch size')
flags.DEFINE_string('meta_batch_size', '40', 'meta batch size')
flags.DEFINE_integer('meta_iter', 300, 'meta_iter')
flags.DEFINE_string('meta_method', 'trpo', '{vpg|trpo|chaser|reptile}')
flags.DEFINE_integer('taskvar', 1, '0 for fwd/bwd, 1 for goal vel (kind of), 2 for goal pose')
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
fast_lr = float(FLAGS.fast_lr)
meta_step_size = float(FLAGS.meta_step_size)
fast_batch_size = int(FLAGS.fast_batch_size)
meta_batch_size = int(FLAGS.meta_batch_size)
meta_iter = FLAGS.meta_iter
meta_method = FLAGS.meta_method
taskvar = FLAGS.taskvar
mode = FLAGS.mode
load_policy = FLAGS.load_policy

# option
max_path_length = 200
num_grad_updates = 1
num_leader_grad_updates = 2

stub(globals())

# task type
if taskvar == 0:
    env = TfEnv(normalize(AntEnvRandDirec()))
    task_var = 'direc'
elif taskvar == 1:
    env = TfEnv(normalize(AntEnvRand()))
    task_var = 'vel'
elif taskvar == 2:
    env = TfEnv(normalize(AntEnvRandGoal()))
    task_var = 'pos'

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

exp_prefix_str = maml_type + '-'+meta_method+'-ant' + task_var + '-' + str(max_path_length)
exp_name_str = maml_type+'_M'+str(num_particles)+svpg_str+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_lr)  + '_mlr_' + str(meta_step_size) + '_randomseed' + str(random_seed)

if load_policy:
    previous_work_path = os.path.join('data',mode,exp_prefix_str,exp_name_str)
    itr_list = []
    for filename in os.listdir(previous_work_path):
        if 'pkl' in filename:
            itr_list.append(int(filename.split('.')[0].split('_')[1]))
    itr_list.sort()
    max_itr = itr_list[-1]
    load_policy = previous_work_path+'/itr_'+str(max_itr)+'_'
    f = open(previous_work_path+'/progress.csv','r')
    progress_lines = [line for line in f.readlines()]
    f.close()
    f = open(previous_work_path+'/progress.csv','w')
    for i in range(max_itr+1):
        f.writelines(progress_lines[i])
    f.close()
else:
    load_policy = None

from rllab.misc.instrument import VariantGenerator, variant

class VG(VariantGenerator):
    @variant
    def fast_lr(self):
        return [fast_lr]
    @variant
    def meta_step_size(self):
        return [meta_step_size]
    @variant
    def fast_batch_size(self):
        return [fast_batch_size]
    @variant
    def meta_batch_size(self):
        return [meta_batch_size]
    @variant
    def seed(self):
        return [random_seed]
    @variant
    def task_var(self):  
        return [taskvar]

v = VG().variants()[0]

# policies
if load_policy is None:
    policy_list = [BMAMLGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=fast_lr,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
        particle_idx=n)
        for n in range(num_particles)]
else:
    policy_list = [0]*num_particles

# baseline
baseline_list = [LinearFeatureBaseline(env_spec=env.spec) for n in range(num_particles)]

# meta learning methods
if meta_method == 'chaser':
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
elif meta_method == 'trpo':
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
else:
    raise ValueError  # {trpo|chaser|reptile}

# run
run_experiment_lite(
    algo.train(),
    exp_prefix=exp_prefix_str,
    exp_name=exp_name_str,
    # Number of parallel workers for sampling
    n_parallel=num_parallel,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="gap",
    snapshot_gap=25,
    sync_s3_pkl=True,
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=random_seed,
    mode=mode,
    #mode="ec2",
    variant=v,
    # plot=True,
    # terminate_machine=False,
    )
