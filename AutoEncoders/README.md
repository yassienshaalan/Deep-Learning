# AutoEncoders
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
##### Here, we show how to implement and test a deep autoencoder. 
### 1. Getting Started
#### - Sample code to run DTOPS on all data (Iteratively cleaning training set, train LSTM, then run RVAE)
```python
    from DeepAE import *
    import time
    import os

    os.chdir("../../")
    x = np.load(r"./data/data.npk")#Load simple MINST Data
    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess=sess, input_dim_list=[784, 225, 100])
        error = ae.fit(x, sess=sess, learning_rate=0.01, batch_size=500, iteration=500, verbose=True)
        R = ae.getRecon(x, sess=sess)
        print("size 100 Runing time:" + str(time.time() - start_time) + " s")
        error = ae.fit(R, sess=sess, learning_rate=0.01, batch_size=500, iteration=500, verbose=True)
```
