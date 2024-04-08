# This is a project for [AMICA](https://amica.scu.edu/) and CSEN342 Deep Learning at Santa Clara University <br>
This work improves upon the benchmark dataset [MR2 by Xu et al., in proceedings of SIGIR 2023](https://dl.acm.org/doi/pdf/10.1145/3539618.3591896) <br>
The best model produced from running `run.py` gives 94 accuracy on the shuffled Chinese dataset testset. <br>
In `data`, we include 3 sets of json objects used to access the actual data. To run this code, plz go check their paper for the github link, and in their github find the dataset google drive link and after downloading put train, val, test folders into the `data` folder of this repo. <br>
Our best model weights are not included because each of them takes 120MB, but please feel free to ask jnian@scu.edu and I will send it to you 
# To train 
Simply go to `run.py` and tweak the parameters to your liking, then just run it. You can also check TensorBoard progress using the file in "exp" folder. Our best model is obtained after 7 epochs on the shuffled Chinese dataset <br>
To inference, simple go to `inference.py` and use a existing model weight to load the appropriate model to run. The lines about data loaders tells which set you are inferencing on. 
