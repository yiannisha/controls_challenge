# Steering Control using Bayesian optimization

## Usage
Follow the task's original instructions - just remember to use the `pid_optim_base` controller.

## Methodology
I used the scripts in the `pid_optimization` directory to explore the solution space for potential
parameters that would optimize the original PID controller.

I used the given cost functions - implemented in `tinyphysics.py` and a custom rollout function `run_rollout_optim` to get the average cost per route for each solution.

I am using a collection of random routes per iteration based on the batch size.

During "training" I played with different values for the batch size and the bounds for the parameters.

I managed to find the optimal parameters using batch size 4 and the bounds I have set in `pid_optimization/optimize.py`.

You can also find another loss function I played around with. I tried "training" the PID on the data that already had a steer command. I then realized that those steer commands were produced by the originally given PID controller so I stopped.

## Tips for optimizing
Some optimizing tips based on my experience:

- `n_iter` works best at around 2-3*`init_points`
- if one locally optimal solution is overfitted then all next solutions will be as well.
- watch the optimizer run live and periodically run an actual evaluation using the best locally optimal solutions, this will help you understand when the optimizer has actually started overfitting - meaning there's no point in keep going.
- if the best solution (even if not overfitted) contains parameters that are exactly equal to the upper or lower bound then there's (most probably) some gains to be made if you increase the corresponding boundary.
- Larger batch size does not equal better convergence - at least for a low number of iterations. I suggest that you always start with a very low number for batch size (i.e. `batch_size=1`) in order to overfit (just like Andrej Karpathy sugggests here[https://karpathy.github.io/2019/04/25/recipe/] for training neural nets). Then when trying a larger number (I tried `batch_size=20`) you can get a feeling if a lower batch size can be better - none of the experiments I did with `batch_size=20` could outperform the results from `batch_size=1`.

## Next Steps
I did this experiment trying to reach the best possible results using bayesian optimization in order to use the optimized PID along a trained RNN to even further decrease the total cost.

I am currently working on the RNN part - I will probably push all further updates on this repo so keep it in mind if you want to see the whole system.

# Comma Controls Challenge!
![Car](./imgs/car.jpg)

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.


## Geting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual routes with actual car and road states.

```
# download necessary dataset (~0.6G)
bash ./download_dataset.sh

# install required packages
# recommended python==3.11
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid 
```

There are some other scripts to help you get aggregate metrics: 
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. It's inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`) and a steer input (`steer_action`) and predicts the resultant lateral acceleration of the car.


## Controllers
Your controller should implement a new [controller](https://github.com/commaai/controls_challenge/tree/master/controllers). This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.


## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(actual\\_lat\\_accel - target\\_lat\\_accel)^2}{steps} * 100$

- `jerk_cost`: $\dfrac{\Sigma((actual\\_lat\\_accel\_t - actual\\_lat\\_accel\_{t-1}) / \Delta t)^2}{steps - 1} * 100$

It is important to minimize both costs. `total_cost`: $(lataccel\\_cost * 50) + jerk\\_cost$

## Submission
Run the following command, and submit `report.html` and your code to [this form](https://forms.gle/US88Hg7UR6bBuW3BA).

```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller pid
```

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma
Like this sort of stuff? You might want to work at comma!
https://www.comma.ai/jobs
