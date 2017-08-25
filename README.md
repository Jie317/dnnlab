### DNNLab

An end-to-end framework for in-production experiments on Deep Learning models for binary classification task on sequetial data.


#### Usage on a complete new machine

- Install CUDA and cudnn.

- Build `tensorflow-gpu` from source.

- Install Dnnlab:

```shell
sudo apt-get update 
sudo apt-get install python3-pip
# checkout this repository
# in the root folder, run
sudo python3 setup.py install
sudo pip3 install -r requirements.txt
```

- Run `python3 example.py -h` under the folder `examples/` to check the usage of arguments. Try this to run an example: `python3 example.py --dc dataset_samples/-3-1-1 -m mlp --fmt csv --mf 10000 --label label`



#### Main Features
- It allows to run huge data on small memory (e.g., RAM < 12G). There is no limit on how many data can be used to train the model, with the implementation of several generator functions. At present, there are only two formats of data files are supported, i.e., CSV and parquet.

- It provides two main arguments, through which user-defined functions can be passed to pre-process data and build neural networks. Data preprocessing and model construction are two core steps to tune DNN models. By passing user-defined functions, users have full control over how the data should be processed before being fed to model, and how the neural layers is built.

- A set of built-in functions can automatically calculate most prevailing evaluation metrics for binary classification models, such as F-measure, Logloss (Logarithmic loss), AUROC (Area Under the Receiver Operating Characteristic curve), Precision-recall curve, and etc. The metric scores for those who have only scalar values will be recorded in a csv file, while other metrics will be registered as pictures or Excel files.

- TensorBoard, a tool to visualize training phase, is adapted to track train loss by batches of sample. This fine-grained visualization also allows to check model's performance on validation data within each training epoch. When dealing with too large data, this may help to find early stop point within the epoch and save training time. 

- In production mode, trained models can be directly exported in the format that works in TensorFlow c++ API and TensorFlow Serving\footnote{A flexible, high-performance serving system for machine learning models built with TensorFlow, designed for production environments.}. Besides, built-in interfaces allows easy implementations for uploading logs and monitoring metrics.


