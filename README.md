# Adverarial latent parameters
Parameterized data set optimization to evaluate limits of neural models and mechanisms
Requires python 3

# Build and activate a conda enviroment
conda env create -f environment.yml adv
source activate adv

# Access the psql DB
- psql adv -h 127.0.0.1 -d adv

# Initialize and add experiments to DB
python db_tools.py --exp=experiments/psvrt.yaml --init

# Reset gradient records and add gradient experiments to DB
python db_tools.py --grads --exp=experiments/gradients.yaml --reset_grads

# Generate example from learned parameters for sampling from a gan
CUDA_VISIBLE_DEVICES=5 python generate_samples.py --param_path=results/biggan_resnet18_2020-01-10_18:28:36_final_params.npz

# Interpolate samples from original to new params
CUDA_VISIBLE_DEVICES=-1 python generate_interpolation.py --param_path=results/biggan_resnet50_2020-01-16-16_39_13_final_params.npz --cat=gorilla --steps=10 --n=5
CUDA_VISIBLE_DEVICES=-1 python generate_interpolation.py --param_path=results/biggan_resnet50_2020-01-16-16_39_13_final_params.npz --cat=fly --steps=10 --n=5

