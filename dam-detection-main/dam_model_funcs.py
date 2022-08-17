import tensorflow as tf
import numpy as np
import DamDataGenerator as data
import pickle
import os
#import focalloss_funcs as floss
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping



save_dir='./nets/'
loss_dir='./nets/losses/'


def build_resnet(num_channels=[16,32,64],num_blocks=[2,2,2],l2_reg=None,reg=None,inits='glorot_uniform',last_layer=True, input_shape=(256,256,4),fl_size=7,dense=128):
    inputs = tf.keras.Input(shape=input_shape)
    if isinstance(inits,float):
        weight_inits = tf.keras.initializers.TruncatedNormal(stddev=inits)
    else:
        weight_inits = inits
    if l2_reg is not None:
        l2_reg = tf.keras.regularizers.l2(l2_reg)
    for i, num_ch in enumerate(num_channels):
        #Downscale
        if i == 0:  #pre7  fl_size was 7,at 7 5
            conv_out = tf.keras.layers.Conv2D(num_ch,fl_size,padding='same',input_shape=input_shape, data_format="channels_last",kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(inputs)
        else:
            conv_out = tf.keras.layers.Conv2D(num_ch,3,padding='same',kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(conv_out)
        conv_out = tf.keras.layers.MaxPool2D(pool_size=[3,3],padding='same',strides=[2,2],data_format="channels_last")(conv_out)

        #Residual blocks
        for j in range(num_blocks[i]):
              #with tf.variable_scope('residual_%d_%d' % (i,j)):
              block_input = conv_out
              conv_out = tf.keras.layers.ReLU()(conv_out)
              conv_out = tf.keras.layers.Conv2D(num_ch,3,padding='same',kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(conv_out)
              conv_out = tf.keras.layers.ReLU()(conv_out)
              conv_out = tf.keras.layers.Conv2D(num_ch, 3, padding='same',kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(conv_out)
              conv_out += block_input
    conv_out = tf.keras.layers.ReLU()(conv_out)
    flattened = tf.keras.layers.Flatten()(conv_out)
    if not last_layer:
        return inputs, flattened
    if reg is None:
        conv_out = tf.keras.layers.Dense(dense,use_bias=True,activation='relu',kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(flattened)  
    else:
        conv_out = tf.keras.layers.Dense(dense,use_bias=True,activation='relu',kernel_initializer=weight_inits, kernel_regularizer=l2_reg, activity_regularizer=tf.keras.regularizers.l1(reg))(flattened)  
    return inputs, conv_out

def build_fullyconv(num_channels=[16,16,32,64],num_blocks=[1,1,1,1],l2_reg=None,reg=None,inits='glorot_uniform',last_layer=True, input_shape=(None,None,4),fl_size=3,dense=128):
    #will still train with 266x266 as the output of data generator, but can run test images one at a time with dif sizes
    inputs = tf.keras.Input(shape=input_shape) 
    if isinstance(inits,float):
        weight_inits = tf.keras.initializers.TruncatedNormal(stddev=inits)
    else:
        weight_inits = inits
    if l2_reg is not None:
        l2_reg = tf.keras.regularizers.l2(l2_reg)
    for i, num_ch in enumerate(num_channels):
        #Downscale
        if i == 0:  
            conv_out = tf.keras.layers.Conv2D(num_ch,fl_size,padding='same',input_shape=input_shape, data_format="channels_last",kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(inputs)
        else:
            conv_out = tf.keras.layers.Conv2D(num_ch,3,padding='same',kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(conv_out)
        conv_out = tf.keras.layers.MaxPool2D(pool_size=[3,3],padding='same',strides=[2,2],data_format="channels_last")(conv_out)

        #Residual blocks
        for j in range(num_blocks[i]):
              #with tf.variable_scope('residual_%d_%d' % (i,j)):
              block_input = conv_out
              conv_out = tf.keras.layers.ReLU()(conv_out)
              conv_out = tf.keras.layers.Conv2D(num_ch,3,padding='same',kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(conv_out)
              conv_out = tf.keras.layers.ReLU()(conv_out)
              conv_out = tf.keras.layers.Conv2D(num_ch, 3, padding='same',kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(conv_out)
              conv_out += block_input
    conv_out = tf.keras.layers.ReLU()(conv_out)
    flattened = tf.keras.layers.GlobalAveragePooling2D()(conv_out)
    if not last_layer:
        return inputs, flattened
    if reg is None:
        conv_out = tf.keras.layers.Dense(dense,use_bias=True,activation='relu',kernel_initializer=weight_inits,kernel_regularizer=l2_reg)(flattened)  
    else:
        conv_out = tf.keras.layers.Dense(dense,use_bias=True,activation='relu',kernel_initializer=weight_inits, kernel_regularizer=l2_reg, activity_regularizer=tf.keras.regularizers.l1(reg))(flattened)  
    return inputs, conv_out



def add_FC_layer(outputs,units,activation=None,use_bias=False,inits='glorot_uniform'):
    if isinstance(inits,float):
        weight_inits = tf.keras.initializers.TruncatedNormal(stddev=inits)
    else:
        weight_inits = inits
    outputs = tf.keras.layers.Dense(units,activation=activation,kernel_initializer=weight_inits,use_bias=use_bias)(outputs)
    return outputs 


def get_optimizer(opt,learning_rate_fn):    
    if opt == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate_fn)
    elif opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    elif opt == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_fn)
    elif opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)
    elif opt == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate_fn)
    return optimizer


def binary_classifier_net(model_num,batch_size=256,save=True,epochs=20,l2_reg=None,inits='glorot_unif',reg=None,LR=.001,opt='adam',dense=128,noise_std=.2):

    output_dim = 1
    net_name = 'FC_'+model_num
    act_f = 'sigmoid' 
    loss="binary_crossentropy"; 
    #alpha=.5; gamma=0
    #loss=[floss.binary_focal_loss(alpha=alpha, gamma=gamma)]

    #load pre-trained model if it exists:
    try:
        net_name_load = [f for f in os.listdir(loss_dir) if net_name in f][0][:-10]
        model = tf.keras.models.load_model(save_dir+net_name_load)
        print(model.summary())
        print('Pre-trained model loaded')
        reloaded = True
        net_name = net_name_load
        LR = LR/10 #assuming starting farther in

    except:
        net_name += '_L2'+str(l2_reg)+'_IN'+str(inits)+'_ar'+str(reg)+'_lr'+str(LR)+'_OP'+str(opt) + '_Ns'+str(noise_std) #+'_al'+str(alpha)+'_gm'+str(gamma)+'_'
        reloaded = False
        #inputs, outputs = build_resnet(l2_reg=l2_reg,inits=inits,reg=reg,num_channels=[32,32,64,64,32],num_blocks=[1,2,2,2,1], fl_size=3,dense=dense) 
        inputs, outputs = build_fullyconv(l2_reg=l2_reg,inits=inits,reg=reg,num_channels=[16,16,32,32,64], num_blocks=[1,1,1,1,1]) 
        outputs = add_FC_layer(outputs,output_dim,activation=act_f,use_bias=True,inits=inits) 
        model = tf.keras.Model(inputs, outputs)
        print(model.summary())


    # setting up training
    DataGen = data.DataGenerator(batch_size,noise_std=noise_std)
    val_im, val_lab = DataGen.get_valbatch()

    #learning_rate_fn = LR
    #can use schedule after looking at initial curves
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10*(DataGen.batch_per_epoch),20*(DataGen.batch_per_epoch)], [LR,LR/10,LR/100])

    optimizer = get_optimizer(opt,learning_rate_fn)
    model.compile(optimizer=optimizer, loss=loss)


    print('Training Supervised Network: ')
    print(net_name)

    csv_logger = CSVLogger(loss_dir+net_name+'Losses.csv', append=True, separator=';')
    if not reloaded:
        first_val = model.evaluate(val_im,val_lab,verbose=1)
        np.save(save_dir+'firstvals/'+net_name+'Firstval.npy',first_val)

    #train model
    early_stopping = EarlyStopping(monitor='val_loss',patience=1)
    model.fit(DataGen,verbose=2,epochs=epochs,steps_per_epoch=DataGen.batch_per_epoch,validation_data=(val_im,val_lab), validation_steps=None,validation_freq=1, callbacks=[csv_logger]) #early_stopping

    if save:
        model.save(save_dir+net_name)


