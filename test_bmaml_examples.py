import os

test_scripts = [
                'python point/bmaml_point.py --num_particles=3 --num_parallel=2 --random_seed=1 --fast_lr=0.1 --meta_step_size=0.01 --fast_batch_size=2 --meta_batch_size=2 --meta_iter=1 --meta_method=trpo --method=vpg --svpg_alpha=1.0',
                'python point/bmaml_point.py --num_particles=3 --num_parallel=2 --random_seed=1 --fast_lr=0.1 --meta_step_size=0.01 --fast_batch_size=2 --meta_batch_size=2 --meta_iter=1 --meta_method=trpo --method=svpg --svpg_alpha=1.0',
                'python point/bmaml_point.py --num_particles=3 --num_parallel=2 --random_seed=1 --fast_lr=0.1 --meta_step_size=0.01 --fast_batch_size=2 --meta_batch_size=2 --meta_iter=1 --meta_method=reptile --method=vpg --svpg_alpha=1.0',
                'python point/bmaml_point.py --num_particles=3 --num_parallel=2 --random_seed=1 --fast_lr=0.1 --meta_step_size=0.01 --fast_batch_size=2 --meta_batch_size=2 --meta_iter=1 --meta_method=chaser --method=svpg --svpg_alpha=1.0',
                ]

for ts_script in test_scripts:
    print(ts_script)
    os.system(ts_script)

print("Done")
