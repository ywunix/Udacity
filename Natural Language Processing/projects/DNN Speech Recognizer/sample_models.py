from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, Dropout,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def final_cnn_output_length(input_length, filter_size, border_mode, stride,
                            dilation=1, pooling=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
        pooling (int): # of Maxpooling layer
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // (stride * pooling)

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    
    tmp_data = input_data
    
    for i in range(recur_layers):
        # Add the first RNN
        rnn = GRU(units, activation='relu',
            return_sequences=True, implementation=2, name=f'rnn_{i}')(tmp_data)
        # Add batch normalization
        tmp_data = BatchNormalization(name=f'bn_rnn_{i}')(rnn)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(tmp_data)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
                return_sequences=True, implementation=2, name='bidir_rnn'))(input_data)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_bidirectional_rnn_model(input_dim, units, filters, kernel_size, conv_stride, conv_border_mode, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    
    '''
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
                return_sequences=True, implementation=2, name='bidir_rnn'))(bn_cnn)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_bidir_rnn')(bidir_rnn)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_cnn)
    '''
    
    # Add the first RNN
    rnn_1 = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn_1')(bn_cnn)
    # Add batch normalization
    bn_rnn_1 = BatchNormalization(name='bn_rnn_1')(rnn_1)
    
    # Concatenate another RNN after rnn_1
    rnn_2 = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn_2')(bn_rnn_1)
    bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(rnn_2)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_2)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model(input_dim = 161,
                cnn_layers = 1,
                filters=256,
                kernel_size=11, 
                conv_stride=2,
                conv_border_mode='same',
                pool_size = 6,
                rnn_layers = 1,
                units=256,
                drop_out_rate = 0.1,
                output_dim=29):
    """ Build a deep network for speech 
    """
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    input_cnn = input_data
    
    # Add CNN layers based on cnn_layers
    for i in range(cnn_layers):
        # Add convolution layer
        conv_1d = Conv1D(filters,
                         kernel_size, 
                         strides=conv_stride, 
                         padding=conv_border_mode,
                         activation='relu',
                         name=f'conv1d_{i}')(input_cnn)
        # Add batch normalization
        bn_cnn = BatchNormalization(name=f'bn_cnn_{i}')(conv_1d)
        # Apply dropout after convolution
        input_cnn = Dropout(drop_out_rate, name=f'dropout_cnn_{i}')(bn_cnn)
        
    # Add Max pooling
    max_pool = MaxPooling1D(pool_size,
                            #strides=conv_stride,
                            #padding=conv_border_mode,
                            name=f'max_pool_cnn_{i}')(input_cnn)
    
    input_rnn = max_pool
    for i in range(rnn_layers):

        # Add the RNN
        bd_rnn = Bidirectional(GRU(units, activation='relu',
                                   recurrent_dropout=drop_out_rate,
                                   return_sequences=True,
                                   implementation=2,
                                   name=f'bd_rnn_{i}'))(input_rnn)
        # Add batch normalization
        bn_rnn = BatchNormalization(name=f'bn_rnn_{i}')(bd_rnn)
        # Apply dropout after Bidirectional RNN
        input_rnn = Dropout(drop_out_rate, name=f'dropout_rnn_{i}')(bn_rnn)
        
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(input_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    model.output_length = lambda x: final_cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, pooling=pool_size)
    
    print("Drop out rate:", drop_out_rate)
    print(model.summary())
    return model