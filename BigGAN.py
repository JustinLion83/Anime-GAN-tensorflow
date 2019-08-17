from layers import *
import numpy as np
import time
from utils import util
import os

'''創建BIGGAN_model類對象'''
class BIGGAN_model(object):
    #------------------------------------------------------------------------------------
    def __init__(self, args):
        self.args = args     # 由argsparser解析過的參數來決定!
        self.d_loss_log = []
        self.g_loss_log = []
        self.layer_num = int(np.log2(self.args.img_size[0])) - 3

        '''inputs'''
        # 控制是否為訓練階段
        self.is_training = tf.placeholder_with_default(False, (), name='is_training')
        
        # 圖
        self.inputs = tf.placeholder(tf.float32,
                                     [None, self.args.img_size[0], self.args.img_size[1], self.args.img_size[2]],
                                     name='inputs')
        # 噪聲
        self.z = tf.placeholder(tf.float32, [None, 1, 1, self.args.z_dim], name='z')  # noise

        # output of D for "Real images"
        real_logits = self.discriminator(self.inputs)

        # output of D for "Fake images"
        self.fake_images = self.generator(self.z)
        fake_logits = self.discriminator(self.fake_images, reuse=True)

        # get loss for discriminator
        self.d_loss = self.discriminator_loss(d_logits_real=real_logits, d_logits_fake=fake_logits)

        # get loss for generator
        self.g_loss = self.generator_loss(d_logits_fake=fake_logits)

        # 找出G跟D可訓練的參數, 個別訓練
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # global step(計算跑了幾次)
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.add_step = self.global_step.assign(self.global_step + 1) # assign:類似於 =

        # 設置指數衰減的學習率(可自訂第幾步開始衰減)
        '''
        在Tensorflow中，為解決設定學習率(learning rate)問題，提供了指數衰減法來解決。
        通過tf.train.exponential_decay函數實現指數衰減學習率。
        
        步驟:
        1.首先使用較大學習率(目的：為快速得到一個比較優的解)
        2.然後通過迭代逐步減小學習率(目的：為使模型在訓練後期更加穩定)
        
        學習率會按照以下公式變化：
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        '''
        self.d_lr = tf.train.exponential_decay(self.args.d_lr,
                                               tf.maximum(self.global_step - self.args.decay_start_steps, 0),
                                               self.args.decay_steps,
                                               self.args.decay_rate)
        
        self.g_lr = tf.train.exponential_decay(self.args.g_lr,
                                               tf.maximum(self.global_step - self.args.decay_start_steps, 0),
                                               self.args.decay_steps,
                                               self.args.decay_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            '''compute_gradients: minimize()的第一步，返回(gradient, variable)对的list'''
            '''apply_gradients  : minimize()的第二步，返回一個執行梯度更新的ops'''
            d_grads = tf.gradients(self.d_loss, d_vars)
            d_opt = tf.train.AdamOptimizer(self.d_lr, beta1=self.args.beta1, beta2=self.args.beta2)
            self.train_d = d_opt.apply_gradients(zip(d_grads, d_vars))

            g_grads = tf.gradients(self.g_loss, g_vars)
            g_opt = tf.train.AdamOptimizer(self.g_lr, beta1=self.args.beta1, beta2=self.args.beta2)
            self.train_g = g_opt.apply_gradients(zip(g_grads, g_vars))
            
            '''
            tf.train.ExponentialMovingAverage這個函數用於更新參數
            採用滑動平均的方法更新參數。這個函數初始化需要提供一個衰減速率（decay）用於控制模型的更新速度。
            這個函數還會維護一個影子變量（也就是更新參數後的參數值）
            這個影子變量的初始值就是這個變量的初始值，影子變量值的更新方式如下：
            shadow_variable = decay * shadow_variable + (1-decay) * variable
            shadow_variable是影子變量，variable表示待更新的變量，也就是變量被賦予的值
            decay為衰減速率。decay一般設為接近於1的數（0.99,0.999）。
            decay越大模型越穩定，因為decay越大，參數更新的速度就越慢，趨於穩定。
            
            apply方法會為每個變量（也可以指定特定變量）創建各自的shadow variable，即影子變量。
            之所以叫影子變量，是因為它會全程跟隨訓練中的模型變量。
            
            tf.train.ExponentialMovingAverage這個函數還提供了自動更新decay的計算方式：
            decay = min（decay，（1+steps）/（10+steps））
            steps是迭代的次數，可以自己設定。作用讓decay變成動態的，訓練前期的值小，後期的值大
            
            使用MovingAverage的三個要素。
            
            1.指定decay參數創建實例：  
            ema = tf.train.ExponentialMovingAverage(decay=0.9999, num_updates = step)
            
            2.對模型變量使用apply方法：  
            maintain_averages_op = ema.apply([var0, var1])
            
            3.在優化方法使用梯度更新模型參數後執行MovingAverage
            with tf.control_dependencies([opt_op]):
                    training_op = tf.group(maintain_averages_op)
                    
            # 其中，tf.group將傳入的操作捆綁成一個操作
            
            訓練時若使用了ExponentialMovingAverage，在保存checkpoint時
            不僅僅會保存模型參數，優化器參數（如Momentum），還會保存ExponentialMovingAverage的shadow variable。
            之前，我們可以直接使用以下代碼restore模型參數，但不會利用ExponentialMovingAverage的結果：
            saver = tf.Saver()
            saver.restore(sess, save_path)
            
            若要使用ExponentialMovingAverage保存的參數：
            variables_to_restore = ema.variables_to_restore()
            saver                = tf.train.Saver(variables_to_restore)
            saver.restore(sess, save_path)
            '''
            # EMA for generator
            with tf.variable_scope("EMA_Weights"):
                if self.args.ema_decay is not None:
                    self.var_ema = tf.train.ExponentialMovingAverage(self.args.ema_decay, num_updates=self.global_step)
                    with tf.control_dependencies([self.train_g]):
                        self.ema_train_g = self.var_ema.apply(tf.trainable_variables(scope='generator'))
                    # assign ema weights
                    self.assign_vars = []
                    for var in tf.trainable_variables(scope='generator'):
                        v = self.var_ema.average(var)
                        if v is not None:
                            self.assign_vars.append(tf.assign(var, v))
                            
    '''使用hinge loss, 避免D太過自信'''
    def discriminator_loss(self, d_logits_real, d_logits_fake):
        # 鼓勵d_logits_real的給分接近 1.0, 不鼓勵給分超過 1.0, 因為會被relu歸零    
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - d_logits_real))
        
        # 鼓勵d_logits_fake的給分接近-1.0, 不鼓勵給分低於-1.0, 因為會被relu歸零
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + d_logits_fake))
        
        loss = real_loss + fake_loss
        return loss

    def generator_loss(self, d_logits_fake):
        '''得分越高越好, 加油!(負號是因為要用minimize的形式~)'''
        loss = -tf.reduce_mean(d_logits_fake)
        return loss
    
    '''生成器'''
    def generator(self, z, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            ch = self.args.g_filters
            x = spectral_deconv2d(z, filters=ch // 2, kernel_size=4, stride=1, is_training=self.is_training,
                                  padding='VALID',
                                  use_bias=False, scope='deconv2d')

            x = ResBlockUp(x, ch // 2, self.is_training, scope='residual_1', reuse=reuse)  # 8*8
            ch = ch // 2

            x = ResBlockUp(x, ch // 2, self.is_training, scope='residual_2', reuse=reuse)  # 16*16
            ch = ch // 2

            x = ResBlockUp(x, ch // 2, self.is_training, scope='residual_3', reuse=reuse)  # 32*32
            ch = ch // 2

            x = attention(x, ch, is_training=self.is_training, scope="attention", reuse=reuse)  # 32*32*128

            x = ResBlockUp(x, ch // 2, self.is_training, scope='residual_4', reuse=reuse)  # 64*64
            ch = ch // 2

            x = ResBlockUp(x, ch // 2, self.is_training, scope='residual_5', reuse=reuse)  # 128*128
            x = batch_norm(x, is_training=self.is_training)
            x = tf.nn.leaky_relu(x)
            x = spectral_conv2d(x, filters=self.args.img_size[2], kernel_size=3, stride=1,
                                is_training=self.is_training,
                                use_bias=False, scope='G_logit')
            x = tf.nn.tanh(x)

            return x

    '''鑑別器'''
    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = self.args.d_filters
            x = ResBlockDown(x, ch * 2, self.is_training, scope='residual_1', reuse=reuse)  # 64*64
            ch = ch * 2

            x = ResBlockDown(x, ch * 2, self.is_training, scope='residual_2', reuse=reuse)  # 32*32
            ch = ch * 2

            x = attention(x, ch, is_training=self.is_training, scope="attention", reuse=reuse)  # 32*32*128

            x = ResBlockDown(x, ch * 2, self.is_training, scope='residual_3', reuse=reuse)  # 16*16
            ch = ch * 2

            x = ResBlockDown(x, ch * 2, self.is_training, scope='residual_4', reuse=reuse)  # 8*8
            ch = ch * 2

            x = ResBlockDown(x, ch * 2, self.is_training, scope='residual_5', reuse=reuse)  # 4*4

            x = spectral_conv2d(x, filters=1, kernel_size=4, padding='VALID', stride=1, is_training=self.is_training,
                                use_bias=False,
                                scope='D_logit')
            x = tf.squeeze(x, axis=[1, 2])

            return x
    
    '''將pixel_values縮放到[-1.0. 1.0]'''
    def preprocess(self, x):
        x = x / 127.5 - 1
        return x

    def train_epoch(self, sess, saver, train_next_element, i_epoch, n_batch, truncated_norm, z_fix=None):
        t_start = None
        global_step = 0
        
        for i_batch in range(n_batch):
            if i_batch == 1:
                t_start = time.time()
            batch_imgs = sess.run(train_next_element)
            batch_imgs = self.preprocess(batch_imgs)
            batch_z = truncated_norm.rvs([self.args.batch_size, 1, 1, self.args.z_dim])
            feed_dict_ = {self.inputs: batch_imgs,
                          self.z: batch_z,
                          self.is_training: True}
            
            # update D network
            _, d_loss, d_lr, g_lr = sess.run([self.train_d, self.d_loss, self.d_lr, self.g_lr], feed_dict=feed_dict_)
            self.d_loss_log.append(d_loss)

            # update G network
            g_loss = None
            if i_batch % self.args.n_critic == 0:
                if self.args.ema_decay is not None:
                    _, g_loss, _, global_step = sess.run(
                        [self.ema_train_g, self.g_loss, self.add_step, self.global_step], feed_dict=feed_dict_)
                else:
                    _, g_loss, _, global_step = sess.run([self.train_g, self.g_loss, self.add_step, self.global_step],
                                                         feed_dict=feed_dict_)
            self.g_loss_log.append(g_loss)

            last_train_str = "[epoch:%d/%d, global_step:%d] -d_loss:%.3f - g_loss:%.3f -d_lr:%.e -g_lr:%.e" % (
                i_epoch + 1, int(self.args.epochs), global_step, d_loss, g_loss, d_lr, g_lr)
            
            if i_batch > 0:
                last_train_str += (' -ETA:%ds' % util.cal_ETA(t_start, i_batch, n_batch))
                
            if (i_batch + 1) % 20 == 0 or i_batch == 0:
                tf.logging.info(last_train_str)

            # show fake_imgs
            if global_step % self.args.show_steps == 0:
                tf.logging.info('generating fake imgs in steps %d...' % global_step)
                # do ema
                if self.args.ema_decay is not None:
                    # save temp weights for generator
                    saver.save(sess, os.path.join(self.args.checkpoint_dir, 'temp_model.ckpt'))
                    sess.run(self.assign_vars, feed_dict={self.inputs: batch_imgs,
                                                          self.z: batch_z,
                                                          self.is_training: False})
                    tf.logging.info('After EMA...')

                if z_fix is not None:
                    show_z = z_fix
                else:
                    show_z = truncated_norm.rvs([self.args.batch_size, 1, 1, self.args.z_dim])
                    
                fake_imgs = sess.run(self.fake_images, feed_dict={self.z: show_z})
                manifold_h = int(np.floor(np.sqrt(self.args.sample_num)))
                util.save_images(fake_imgs, [manifold_h, manifold_h],
                                 image_path=os.path.join(self.args.result_dir,
                                                         'fake_steps_' + str(global_step) + '.jpg'))
                if self.args.ema_decay is not None:
                    # restore temp weights for generator
                    saver.restore(sess, os.path.join(self.args.checkpoint_dir, 'temp_model.ckpt'))
                    tf.logging.info('Recover weights over...')

        return global_step, self.d_loss_log, self.g_loss_log
