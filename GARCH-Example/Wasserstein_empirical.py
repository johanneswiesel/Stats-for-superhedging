#!/usr/bin/env python3 
"""
@authors: Stephan Eckstein, Johannes Wiesel 
"""

import numpy as np
import tensorflow as tf

def calculate_wasserstein (data, epsilon_const, alpha, g_wass):
    
    tf.reset_default_graph()
    N = data.shape[0]
    d = 1
    DATA = np.reshape(data, (N,d))
    EPS_CONST = 0.5
    EPS_N = EPS_CONST * (N ** (-(1/(2*max(d, 2)))))
    print('EPS_N = ' + str(EPS_N))
    ALPHA = alpha
    print('ALPHA = ' + str(ALPHA))
    SQRT_BUFFER = 10 ** (-6)  # gradients of the root function are undefined at 0, and this can cause numerical issues
    
    N_STEPS = 50000 + d * 10000  # Number of iterations the network is trained: Potentially adjust 
    BATCH_SIZE = 2 ** 7 * (2 ** int(round(np.log2(d))))  # batch size of the network
    GAMMA = d*1000  # penalty factor to enforce the inequality constraint in the dual of wasserstein ball optimization
    
    print('BATCH_SIZE = ' + str(BATCH_SIZE))
    print('GAMMA = ' + str(GAMMA))
    
    def AVAR_f(p, t):
        mt = tf.nn.relu(p-t)/ALPHA
        out = t + mt
        return out
    
    def univ_approx(x, name, hidden_dim=32*d, input_dim=d):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            ua_w = tf.get_variable('ua_w1', shape=[input_dim, hidden_dim],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            ua_b = tf.get_variable('ua_b1', shape=[hidden_dim],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            z = tf.matmul(x, ua_w) + ua_b
            a = tf.nn.relu(z)
            ua_w2 = tf.get_variable('ua_w2', shape=[hidden_dim, hidden_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            ua_b2 = tf.get_variable('ua_b2', shape=[hidden_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            z2 = tf.matmul(a, ua_w2) + ua_b2
            a2 = tf.nn.relu(z2)
            ua_w3 = tf.get_variable('ua_w3', shape=[hidden_dim, hidden_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            ua_b3 = tf.get_variable('ua_b3', shape=[hidden_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            z3 = tf.matmul(a2, ua_w3) + ua_b3
            a3 = tf.nn.relu(z3)
            ua_v = tf.get_variable('ua_v', shape=[hidden_dim, 1],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            ua_b4 = tf.get_variable('ua_b4', shape=[1],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            z = tf.matmul(a3, ua_v) + ua_b4
        return tf.reduce_sum(z, axis=1)
    
    
    x_marg = tf.placeholder(dtype=tf.float32, shape=[None, d])
    x = tf.placeholder(dtype=tf.float32, shape=[None, d])
    x2 = tf.placeholder(dtype=tf.float32, shape=[None, d])
    
    tau_variable = tf.get_variable('tau_variable', shape=[1], initializer=tf.constant_initializer(1),
                                  dtype=tf.float32)
    lambda_variable = tf.get_variable('lambda_variable', shape=[1], initializer=tf.constant_initializer(0.5), dtype=tf.float32)
    H_variable = tf.get_variable('H_variable', shape=[d], initializer=tf.constant_initializer(0), dtype=tf.float32)
    
    h_marg = univ_approx(x_marg, 'lonely')
    h_1 = univ_approx(x, 'lonely')
    diff = AVAR_f(g_wass(x2) - tf.reduce_sum(H_variable * (x2 - 1), axis=1), tau_variable) - h_1 - tf.nn.relu(lambda_variable) * tf.sqrt(tf.reduce_sum(tf.square(x-x2), axis=1))
    obj_fun = tf.nn.relu(lambda_variable) * EPS_N + tf.reduce_mean(h_marg) + GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(diff))) + tf.nn.relu(-lambda_variable)
    
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(obj_fun)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        def gen_points():
            c_gen = int(round(BATCH_SIZE/3))
            while 1:
                p = np.random.choice(np.arange(N), 3 * c_gen)
                data_1 = DATA[p, :]
                data_2 = np.zeros([3 * c_gen, d])
                data_2[:c_gen, :] = data_1[:c_gen, :] + 0.001 * np.random.randn(c_gen, d)
                p2 = np.random.choice(np.arange(N), c_gen)
                data_2[c_gen:2*c_gen, :] = DATA[p2, :]
                p3 = np.random.choice(np.arange(N), c_gen)
                data_2[2*c_gen:, :] = DATA[p3, :]
                yield data_1, data_2
    
    
        gen = gen_points()
        value_list = []
        
        N_STEPS=16000+d*2000
    
        for t in range(1, N_STEPS+1):
            sample1, sample2 = next(gen)
            (c, _, dval) = sess.run([obj_fun, train_op, diff], feed_dict={x: sample1, x2: sample2, x_marg: sample1})
            value_list.append(c)
    
            if t % 2000 == 0:
                (h_val, lam_val, tau_val) = sess.run([H_variable, lambda_variable, tau_variable])
                print(str(t)) 
                print(h_val)
                print(lam_val)
                print(tau_val)
                print(np.mean(value_list[-5000:]))
        final_val = np.mean(value_list[-5000:])
    
    return(final_val)
