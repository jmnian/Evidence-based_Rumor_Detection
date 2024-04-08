Project for CSEN342 Deep Learning at Santa Clara University <br>
This work improves upon the benchmark dataset MR2 by Xu et al., in proceedings of SIGIR 2023 <br>
In here we provided 5 training samples, all are rumor posts. If you'd like to get all the data, please check their paper for the github link for the dataset download link <br>
Our best model weights are not included because each of them takes 120MB of memory, but please feel free to ask jnian@scu.edu and I will send it to you <br>
To train, simply go to run.py and check the parameters are to your liking, then just run it. You can also check TensorBoard progress using the file in "exp" folder <br>
To inference, simple go to inference.py and use a existing model weight to load the appropriate model then run 