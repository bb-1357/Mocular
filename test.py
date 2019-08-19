#coding:utf-8
import tensorflow as tf
import cv2 as cv
import numpy as np
import os
from scipy.io import loadmat
# 设定神经网络的超参数
# 定义神经网络可以接收的图片的尺寸和通道数
IMAGE_SIZE_W = 800
IMAGE_SIZE_H = 352
NUM_CHANNELS = 3
# 定义第一层卷积核的大小和个数，same卷积，后跟池化4X4并且4步
CONV1_SIZE = 3
CONV1_KERNEL_NUM = 64
# 定义第二层卷积核的大小和个数，same卷积,后跟池化2X2并且2步
CONV2_SIZE = 3
CONV2_KERNEL_NUM = 128
# 定义第3层卷积核的大小和个数,sanme卷积，后跟池化2X2并且2步
CONV3_SIZE = 3
CONV3_KERNEL_NUM = 256
# 定义第4层卷积核的大小和个数，same卷积，后跟池化2X2并且2步
CONV4_SIZE = 3
CONV4_KERNEL_NUM = 256
#删除所有的全连接层，参数量过于庞大

# 定义第5层全连接层的神经元个数
#FC_SIZE = 60000
# 定义第6层全连接层的神经元个数
#OUTPUT_NODE = 57600
# 定义初始化网络权重函数


#输出的两个不同大小
SIZE_1 = [176,400]
#resize之后的大小，也是标定的大小，resize之后跟着的四个卷积，其大小都不变跟conv4一样的same卷积。
SIZE_2 = [120,480]

#定义第二个网络的基本参数，输入是不变的
# 定义第一层卷积核的大小和个数，2步
S_CONV1_SIZE = 2      
S_CONV1_KERNEL_NUM = 1
# resize之后，定义后面四层卷积核的大小和个数
S_CONV2_SIZE = 3
S_CONV2_KERNEL_NUM = 1

short_cut1 = np.zeros([20,176,400])
short_cut2 = np.zeros([20,120,480])

'''temp = []
real_temp = np.empty(shape=[20,176,400])
middle = np.empty(shape=[20,176,400])
mid_1 = np.empty(shape=[176,400])


y = []
real_y = np.empty(shape=[20,120,480])
middle_2 = np.empty(shape=[20,176,400])
mid_2 = np.empty(shape=[176,400])'''


# 定义训练过程中的超参数
BATCH_SIZE = 20 # 一个 batch 的数量
LEARNING_RATE_BASE = 0.005 # 初始学习率
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARIZER = 0.0001 # 正则化项的权重
STEPS = 5000 # 最大迭代次数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均的衰减率
MODEL_SAVE_PATH="./model/" # 保存模型的路径
MODEL_NAME="Depth_Model" # 模型命名
# 训练过程
def backward(p1,p2):
    # x, y_是定义的占位符,需要指定参数的类型,维度(要和网络的输入与输出维度一致),类似 于函数的形参,运行时必须传入值
    #x = tf.placeholder(tf.float32,[BATCH_SIZE,forward1.IMAGE_SIZE_H,forward1.IMAGE_SIZE_W,forward1.NUM_CHANNELS])
    x = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE_H,IMAGE_SIZE_W,NUM_CHANNELS])
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%p1")
    print(p1.shape)
    print(type(p1))    
    
    #y_ = tf.placeholder(tf.float32, [20,120,480])
    y_ = tf.placeholder(tf.float32, [20,120,480])
    #y = forward1.forward_1(x,True, REGULARIZER)+forward1.forward_2(x,True,REGULARIZER) # 调用前向传播网络得到维度为  10 的 tensor
    y = forward_1(x,True, REGULARIZER)
    temp = np.empty(shape = [20,120,480],dtype= np.float32)
    '''with tf.Session() as sess:
        tf.initialize_all_variables().run()
        middle = sess.run(y.eval())
        print(type(middle))
        for i in range(0,relu9_shape[0]):            
            mid = cv2.resize(middle[i],(120,480),interpolation=cv.INTER_NEAREST)
            mid_1.append(mid)
    for i in range(20):
        for j in range(120):
            for k in range(480):
                temp[i][j][k]=mid_1[i][j][k]'''  
    real_y =  tf.convert_to_tensor(temp)                 
    
    global_step = tf.Variable(0, trainable=False) # 声明一个全局计数器,并输出化为 0
    # 先是对网络最后一层的输出 y 做 softmax,通常是求取输出属于某一类的概率,其实就是一个 num_classes 大小的向量,
    # 再将此向量和实际标签值做交叉熵,需要说明的是该函数返回的是一个向量
    
    #ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    print(type(y))
    mse = tf.reduce_sum(tf.square(real_y-y_)) # 再对得到的向量求均值就得到 loss
    
    loss = mse + tf.add_n(tf.get_collection('losses')) # 添加正则化中的 losses'''     
    
    # 实现指数级的减小学习率,可以让模型在训练的前期快速接近较优解,又可以保证模型在训练后期不会有太大波动
    
    # 计算公式:decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
    
    learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    1,
    LEARNING_RATE_DECAY,
    staircase=True) # 当 staircase=True 时,(global_step/decay_steps)则被转化为整数,以此来选择不同的衰减方式
   
    # 传入学习率,构造一个实现梯度下降算法的优化器,再通过使用 minimize 更新存储要训练的变量的列表来减小 loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
   
    # 实现滑动平均模型,参数 MOVING_AVERAGE_DECAY 用于控制模型更新的速度。训练过程中会对每一个变量维护一个影子变量,这个影子变量的初始值
    # 就是相应变量的初始值,每次变量更新时,影子变量就会随之更新
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
   
    with tf.control_dependencies([train_step, ema_op]): # 将 train_step 和 ema_op 两个训练操作绑 定到 train_op 上
        train_op = tf.no_op(name='train')    
    with tf.Session() as sess: 
        tf.initialize_all_variables().run()
        saver = tf.train.Saver() # 实例化一个保存和恢复变量的 saver
        print("已经完成实例化")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("已完成所有变量的初始化")
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) # 通过 checkpoint 文件定位到最新保存的模型
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) # 加载最新的模型
        for i in range(STEPS):
            xs, ys = p1,p2   # 读取一个 batch 的数据,这里我直接传入一个batch,一个一个的传,也就10个batch
            print(type(xs))
            print(xs.shape)
            reshaped_xs = np.reshape(xs,(
            # 将输入数据 xs 转换成与网络输入相同形状的矩阵
            20,
            352,
            800,
            3))
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%xs")
            print(reshaped_xs.shape)
            print(type(reshaped_xs))
            
                # 喂入训练图像和标签,开始训练
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:reshaped_xs,y_: ys})
            if i % 100 == 0: # 每迭代 100 次打印 loss 信息,并保存最新的模型
                print("After %d training step(s), loss on training batch is %g." % (step,loss_value))       
                saver.save(sess,
                os.path.join(MODEL_SAVE_PATH,
                MODEL_NAME),
                global_step=global_step)
  

def conv2d(x,w):
    '''
    args:
    x: 一个输入 batch
    w: 卷积层的权重
    '''
    # strides 表示卷积核在不同维度上的移动步长为 1,第一维和第四维一定是 1,这是因为卷积层的步长只对矩阵的长和宽有效;
    # padding='SAME'表示使用全 0 填充,而'VALID'表示不填充
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
def V_conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='VALID')
   # 定义最大池化操作函数
def max_pool_2x2(x):
    '''
    args:
    x: 一个输入 batch
    '''
    # ksize 表示池化过滤器的边长为 2,strides 表示过滤器移动步长是 2,'SAME'提供使用全 0 填充    
    #return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#定义前向传播的过程

def max_pool_4x4(x):
    #return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
def get_weight(shape, regularizer):
    '''
    args:
    shape:生成张量的维度regularizer: 正则化项的权重
    '''
    # tf.truncated_normal 生成去掉过大偏离点的正态分布随机数的张量,stddev 是指定标准差
    #w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    w = tf.Variable(tf.random.truncated_normal(shape,stddev=0.1))
    # 为权重加入 L2 正则化,通过限制权重的大小,使模型不会随意拟合训练数据中的随机噪音
    if regularizer != None: 
        #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
    # 定义初始化偏置项函数
def get_bias(shape):
    '''
    args:
    shape:生成张量的维度
    '''
    b = tf.Variable(tf.zeros(shape))
    # 统一将 bias 初始化为 0
    return b
    # 定义卷积计算函数
def forward_1(x, train, regularizer):
    '''
    args:
    x: 一个输入 batch
    train: 用于区分训练过程 True,测试过程 False
    regularizer:正则化项的权重
    '''

    # 实现第一层卷积层的前向传播过程
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer) # 初始化卷积核
    conv1_b = get_bias([CONV1_KERNEL_NUM]) # 初始化偏置项
    conv1 = conv2d(x, conv1_w) # 实现卷积运算
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) # 对卷积后的输出添加偏置,并过 relu 非线性激活函数
    pool1 = max_pool_4x4(relu1) # 将激活后的输出进行最大池化
    #print("------------------------------------------------------")
    #print(tf.shape(conv1_w ))

    # 实现第二层卷积层的前向传播过程,并初始化卷积层的对应变量
    conv2_w=get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w) # 该层的输入就是上一层的输出 pool1
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)
    #print("------------------------------------------------------")
    # print(tf.shape(conv2_w))

    # 实现第三层卷积层的前向传播过程,并初始化卷积层的对应变量
    conv3_w=get_weight([CONV3_SIZE,CONV3_SIZE,CONV2_KERNEL_NUM,CONV3_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv3_b = get_bias([CONV3_KERNEL_NUM])
    conv3 = conv2d(pool2, conv3_w) # 该层的输入就是上一层的输出 pool1

    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))
    pool3 = max_pool_2x2(relu3)

    #print("------------------------------------------------------")
    #print(tf.shape(conv3_w))

    # 实现第四层卷积层的前向传播过程,并初始化卷积层的对应变量
    conv4_w=get_weight([CONV4_SIZE,CONV4_SIZE,CONV3_KERNEL_NUM,CONV4_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv4 = conv2d(pool3, conv4_w) # 该层的输入就是上一层的输出 pool1
    conv4_b = get_bias([CONV4_KERNEL_NUM])
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_b))

    pool4 = max_pool_2x2(relu4)    
    pool4_shape = pool4.get_shape().as_list()

    '''with tf.compat.v1.Session() as sess:
        middle = pool4.eval(session=sess)
        for i in range(0,pool4_shape[0]):
            mid_1 = cv.resize(middle(i),(176,400),interpolation=cv.INTER_NEAREST)
            temp.append(mid_1)
    for i in range(20):
        for j in range(176):
            for k in range(400):
                real_temp[i][j][k]=temp[i][j][k] '''     


        #五六七八九层都是一样的方法,此时大小为176*400
    conv5_w=get_weight([CONV4_SIZE,CONV4_SIZE,1,CONV4_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv5 = conv2d(pool4, conv5_w) # 该层的输入就是上一层的输出 pool1
    conv5_b = get_bias([CONV4_KERNEL_NUM])
    relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_b))

    conv6_w=get_weight([CONV4_SIZE,CONV4_SIZE,CONV4_KERNEL_NUM,CONV4_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv6 = conv2d(relu5, conv6_w) # 该层的输入就是上一层的输出 pool1
    conv6_b = get_bias([CONV4_KERNEL_NUM])
    relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_b))

    conv7_w=get_weight([CONV4_SIZE,CONV4_SIZE,CONV4_KERNEL_NUM,CONV4_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv7 = conv2d(relu6, conv7_w) # 该层的输入就是上一层的输出 pool1
    conv7_b = get_bias([CONV4_KERNEL_NUM])
    relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_b))

    conv8_w=get_weight([CONV4_SIZE,CONV4_SIZE,CONV4_KERNEL_NUM,CONV4_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv8 = conv2d(relu7, conv8_w) # 该层的输入就是上一层的输出 pool1
    conv8_b = get_bias([CONV4_KERNEL_NUM])
    relu8 = tf.nn.relu(tf.nn.bias_add(conv8, conv8_b))
    #short_cut1 = relu8

    conv9_w=get_weight([CONV4_SIZE,CONV4_SIZE,CONV4_KERNEL_NUM,CONV4_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv9 = conv2d(relu8, conv9_w) # 该层的输入就是上一层的输出 pool1
    conv9_b = get_bias([CONV4_KERNEL_NUM])
    relu9 = tf.nn.relu(tf.nn.bias_add(conv9, conv9_b))
    #short_cut2 = relu9

    '''with tf.Session() as sess:
        middle_2 = relu9.eval(session=sess)
        for i in range(0,relu9_shape[0]):            
            mid_2 = cv2.resize(middle[i],(120,480),interpolation=cv.INTER_NEAREST)
            y.append(mid_2)
    for i in range(20):
        for j in range(120):
            for k in range(480):
                real_y[i][j][k]=y[i][j][k]'''       
    return relu9
  
#从原图再次卷积得到语义信息
def forward_2(x, train, regularizer):
    '''
    args:
    x: 一个输入 batch
    train: 用于区分训练过程 True,测试过程 False
    regularizer:正则化项的权重
    '''
    # 实现第一层卷积层的前向传播过程
    conv1_w = get_weight([S_CONV1_SIZE, S_CONV1_SIZE, NUM_CHANNELS, S_CONV1_KERNEL_NUM], regularizer) # 初始化卷积核
    conv1_b = get_bias([S_CONV1_KERNEL_NUM]) # 初始化偏置项
    conv1 = V_conv2d(x, conv1_w) # 实现卷积运算
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) # 对卷积后的输出添加偏置,并过 relu 非线性激活函数

    # 实现第二层卷积层的前向传播过程,并初始化卷积层的对应变量
    conv2_w=get_weight([S_CONV2_SIZE,S_CONV2_SIZE,S_CONV1_KERNEL_NUM,S_CONV2_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(relu1, conv2_w) # 该层的输入就是上一层的输出 
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))

    # 实现第二层卷积层的前向传播过程,并初始化卷积层的对应变量
    conv3_w=get_weight([S_CONV2_SIZE,S_CONV2_SIZE,S_CONV1_KERNEL_NUM,S_CONV2_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv3 = conv2d(relu2, conv3_w) # 该层的输入就是上一层的输出
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))+short_cut1

    # 实现第二层卷积层的前向传播过程,并初始化卷积层的对应变量
    conv4_w=get_weight([S_CONV2_SIZE,S_CONV2_SIZE,S_CONV1_KERNEL_NUM,S_CONV2_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv4 = conv2d(relu3, conv4_w) # 该层的输入就是上一层的输出
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_b))+short_cut2
    
    # 实现第二层卷积层的前向传播过程,并初始化卷积层的对应变量
    conv5_w=get_weight([S_CONV2_SIZE,S_CONV2_SIZE,S_CONV1_KERNEL_NUM,S_CONV2_KERNEL_NUM],regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv5 = conv2d(relu4, conv5_w) # 该层的输入就是上一层的输出
    relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_b))
    #pool_shape = relu5.get_shape().as_list()
    y = []
    real_y = np.empty(shape=[20,120,480])
    with tf.Session() as sess:
        middle = relu5.eval(session = sess)
        for i in range(0,20):            
            mid = cv2.resize(middle[i],(120,480),interpolation=cv.INTER_NEAREST)
            y.append(mid)
    for i in range(20):
        for j in range(120):
            for k in range(480):
                real_y[i][j][k]=y[i][j][k]       
    return real_y    

if __name__ == '__main__':
    label = []
    #print(type(label))
    label_folder = '/home/hp209/Mocular/black120480/'
    files =os.listdir(label_folder)
    files.sort()
    for file in files: 
        file_path = os.path.join(label_folder,file)	 
        a = np.loadtxt(file_path)
        a = np.array(a,dtype = np.float32)
        label.append(a)   
    real_label = np.empty(shape=[20,120,480])
    for i in range(20):
        for j in range(120):
            for k in range(480):
                real_label[i][j][k]=label[i][j][k]
    real_label = np.array(real_label,dtype='float32')

    data_folder = '/home/hp209/Mocular/original/mat'
    files =os.listdir(data_folder)
    files.sort()
    data = []
    for file in files: 
        file_path = os.path.join(data_folder, file)
        #print(file_path)
        a =loadmat(file_path)
        a['D'] = np.array(a['D'],dtype = np.float32)
        data.append(a['D'])    
    real_data = np.empty(shape=[20,352,800,3],dtype=np.float32)
    for i in range(20):
        for j in range(352):
            for k in range(800):
                for s in range(3):
                    real_data[i][j][k][s]=data[i][j][k][s]
    real_data = np.array(real_data,dtype = 'float32')
    backward(real_data,real_label) 
