# Baysian Model-Agnostic Meta-Learning

This repository contains implementations of the paper, [Bayesian Model-Agnostic Meta-Learning (Jaesik Yoon and Taesup Kim et al., NuerIPS 2018)](https://arxiv.org/abs/1806.03836). It includes code for running the reinforcement learning task, 2D point navigation task described in the paper.

To comparison with MAML and Ensemble MAML, we implemented EMAML options.
You can run that with the options described as following running scripts.

*NOTE that some rllab source codes are changed for a clean output and this source code is not working with CUDA10. We tested with TensorFlow==v.1.12.0)* 

This code is started from [MAML RL repository](https://github.com/cbfinn/maml_rl).

For the regression experiments, plese see [this repository](https://github.com/jaesik817/bmaml).

## Quick run
This code is based on [rllab](https://github.com/rll/rllab).
You can run this code by installing enclosed requirements.
```
pip install -r requirements
```

Before starting experiments, you can check your environment by running test_bmaml_examples.py.
```
python test_bmaml_examples.py
```

The instructions for the experiements described in the paper are as followed.

* Examples of BMAML running script
```
# SVPG-TRPO with 5 particles 
python bmaml_examples/point/bmaml_point.py --num_particles=5 --num_parallel=10 --fast_lr=0.1 --meta_step_size=0.01 --fast_batch_size=10 --meta_batch_size=20 --meta_iter=100 --meta_method=trpo --method=svpg --svpg_alpha=1.0

# SVPG-Chaser with 5 particles
python bmaml_examples/point/bmaml_point.py --num_particles=5 --num_parallel=10 --fast_lr=0.1 --meta_step_size=0.01 --fast_batch_size=10 --meta_batch_size=20 --meta_iter=100 --meta_method=chaser --method=svpg --svpg_alpha=0.1
```

* Examples of EMAML running script
```
# VPG-TRPO with 5 particles
python bmaml_examples/point/bmaml_point.py --num_particles=5 --num_parallel=10 --fast_lr=0.1 --meta_step_size=0.01 --fast_batch_size=10 --meta_batch_size=20 --meta_iter=100 --meta_method=trpo --method=vpg --svpg_alpha=1.0

# VPG-Reptile with 5 particles
python bmaml_examples/point/bmaml_point.py --num_particles=5 --num_parallel=10 --fast_lr=0.1 --meta_step_size=0.01 --fast_batch_size=10 --meta_batch_size=20 --meta_iter=100 --meta_method=reptile --method=vpg --svpg_alpha=1.0
```

You can plot the results similar to the paper with plotter.py by setting configurations.

## Contact
Any feedback is welcome! Please open an issue on this repository or send email to Jaesik Yoon (jaesik817@gmail.com), Taesup Kim (taesup.kim@umontreal.ca) or Sungjin Ahn (sungjin.ahn@rutgers.edu).

