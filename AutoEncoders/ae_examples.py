from DeepAE import *



##################### test a machine with different data size#####################
def test():
    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess=sess, input_dim_list=[784, 625, 400, 225, 100])
        error = ae.fit(x[:1000], sess=sess, learning_rate=0.01, batch_size=500, iteration=1000, verbose=False)

    print("size 1000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess=sess, input_dim_list=[784, 625, 400, 225, 100])
        error = ae.fit(x[:10000], sess=sess, learning_rate=0.01, batch_size=500, iteration=1000, verbose=False)

    print("size 10,000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess=sess, input_dim_list=[784, 625, 400, 225, 100])
        error = ae.fit(x[:20000], sess=sess, learning_rate=0.01, batch_size=500, iteration=1000, verbose=False)

    print("size 20,000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess=sess, input_dim_list=[784, 625, 400, 225, 100])
        error = ae.fit(x[:50000], sess=sess, learning_rate=0.01, batch_size=500, iteration=1000, verbose=False)

    print("size 50,000 Runing time:" + str(time.time() - start_time) + " s")


if __name__ == "__main__":
    import time
    import os

    os.chdir("../../")
    x = np.load(r"./data/data.npk")
    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess=sess, input_dim_list=[784, 225, 100])
        error = ae.fit(x, sess=sess, learning_rate=0.01, batch_size=500, iteration=500, verbose=True)
        R = ae.getRecon(x, sess=sess)
        print("size 100 Runing time:" + str(time.time() - start_time) + " s")
        error = ae.fit(R, sess=sess, learning_rate=0.01, batch_size=500, iteration=500, verbose=True)
    # test()
