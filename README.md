# This is a project for [AMICA](https://amica.scu.edu/) and CSEN342 Deep Learning at Santa Clara University <br>
This work improves upon the benchmark dataset: <br> [MR2: A Benchmark for Multimodal Retrieval-Augmented Rumor Detection in Social Media by Xu et al., in proceedings of SIGIR 2023](https://dl.acm.org/doi/pdf/10.1145/3539618.3591896) <br>
The best model produced from running `run.py` gives 94 accuracy on the shuffled Chinese dataset testset. <br>

Our best model weights are not included because each of them takes 120MB, but please feel free to ask jnian@scu.edu and I will send it to you 
# To train 
In `data`, we include 4 sets of json objects used to access the actual data. In order to run the training loop, plz go check [MR2 Github](https://github.com/THU-BPM/MR2), find the full dataset google drive link. After downloading, put train, val, test folders into the `data` folder of this repo. <br>
To run the training loop, simply go to `run.py` and tweak the parameters to your liking and run it. You can also check TensorBoard progress using the file in "exp" folder. Our best model is obtained after 7 epochs on the shuffled Chinese dataset <br>
To check tensorboard, do `tensorboard --logdir exp/exp_2024-xxxx`
To inference, simple go to `inference.py` and use a existing model weight to load the appropriate model to run. The lines about data loaders tells which dataset you are inferencing on. 
