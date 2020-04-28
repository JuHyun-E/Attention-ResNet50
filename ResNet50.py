import pprint
from definition import *


def ResNet50(input_shape, classes):

    params = {}

    # X_input = tf.placeholder(tf.float32, shape=input_shape, name='input_layer')

    X = zero_padding(input_shape, (3, 3))
    params['input'] = input_shape
    params['zero_pad'] = X

    # Stage 1, 1 layer
    params['stage1'] = {}
    A_1, params['stage1']['conv'] = conv2d(X, filters=64, k_size=(7, 7), strides=(2, 2), padding='VALID', name='conv1')
    A_1_bn = batch_norm(A_1, name='bn_conv1')
    A_1_act = tf.nn.relu(A_1_bn)
    A_1_pool = tf.nn.max_pool2d(A_1_act, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
    params['stage1']['bn'] = A_1_bn
    params['stage1']['act'] = A_1_act
    params['stage1']['pool'] = A_1_pool

    # Stage 2, 9 layers
    params['stage2'] = {}
    A_2_cb, params['stage2']['cb'] = convolution_block(A_1_pool, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    A_2_ib1, params['stage2']['ib1'] = identity_block(A_2_cb, f=3, filters=[64, 64, 256], stage=2, block='b')
    A_2_ib2, params['stage2']['ib2'] = identity_block(A_2_ib1, f=3, filters=[64, 64, 256], stage=2, block='c')

    # Stage 3, 12 layers
    params['stage3'] = {}
    A_3_cb, params['stage3']['cb'] = convolution_block(A_2_ib2, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    A_3_ib1, params['stage3']['ib1'] = identity_block(A_3_cb, f=3, filters=[128, 128, 512], stage=3, block='b')
    A_3_ib2, params['stage3']['ib2'] = identity_block(A_3_ib1, f=3, filters=[128, 128, 512], stage=3, block='c')
    A_3_ib3, params['stage3']['ib3'] = identity_block(A_3_ib2, f=3, filters=[128, 128, 512], stage=3, block='d')

    # Stage 4, 18 layers
    params['stage4'] = {}
    A_4_cb, params['stage4']['cb'] = convolution_block(A_3_ib3, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    A_4_ib1, params['stage4']['ib1'] = identity_block(A_4_cb, f=3, filters=[256, 256, 1024], stage=4, block='b')
    A_4_ib2, params['stage4']['ib2'] = identity_block(A_4_ib1, f=3, filters=[256, 256, 1024], stage=4, block='c')
    A_4_ib3, params['stage4']['ib3'] = identity_block(A_4_ib2, f=3, filters=[256, 256, 1024], stage=4, block='d')
    A_4_ib4, params['stage4']['ib4'] = identity_block(A_4_ib3, f=3, filters=[256, 256, 1024], stage=4, block='e')
    A_4_ib5, params['stage4']['ib5'] = identity_block(A_4_ib4, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # Stage 5, 9 layers
    params['stage5'] = {}
    A_5_cb, params['stage5']['cb'] = convolution_block(A_4_ib5, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    A_5_ib1, params['stage5']['ib1'] = identity_block(A_5_cb, f=3, filters=[512, 512, 2048], stage=5, block='b')
    A_5_ib2, params['stage5']['ib2'] = identity_block(A_5_ib1, f=3, filters=[512, 512, 2048], stage=5, block='c')

    # Average Pooling
    A_avg_pool = tf.nn.avg_pool2d(A_5_ib2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='avg_pool')  #
    params['avg_pool'] = A_avg_pool

    # Output Layer, 1 layer
    A_flat = flatten(A_avg_pool)
    params['flatten'] = A_flat
    A_out, params['out'] = dense(X=A_flat, out=classes, name='fc'+str(classes))

    return A_out, params


# if __name__ == '__main__':
#     A, params = ResNet50(input_shape=[64, 64, 3], classes=2)
#     pprint.pprint(params, stream=open('ResNet50.json', 'w'), indent=2)
#     print(pprint.pprint(params))
