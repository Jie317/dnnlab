def build_dnn_model(model_name, max_feature, input_length):
    '''Build and compile Keras models.

    # Example

    ```python
        from keras.models import Sequential, Model
        from keras.layers import Dense
        # multilayer perceptrons
        if model_name == 'mlp':
            model = Sequential()
            model.add(Dense(128, activation='relu', input_length=input_length))
            model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['mse'])
        return model
    ```

    # Arguments
        model_name: name of the model to use (you can build serveral models
          and use them by giving them a name in the 'if' condition).
        max_feature: only relevant when using 'Embedding' layer
        input_length: length of input sequences.

    # Return
        Compiled Keras model

    '''

    print('Using built-in build_dnn_model function')
    from keras.models import Sequential
    from keras.layers import Dense

    print('Max feature: %d, input length: %d' % (max_feature, input_length))

    # multilayer perceptrons
    if model_name == 'mlp':
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(input_length, )))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['binary_crossentropy'])

    return model


def launch(args=None,
           dataframe_process=None,
           sep_cols=None,
           build_dnn_model=build_dnn_model,
           send_metric=None):
    import os
    os.path.exists('results/') or os.makedirs('results/')

    if args is None:
        from dnnlab import argument_parser
        args = argument_parser.get_parser().parse_args()
        argument_parser.print_args(args)
    from dnnlab.model_manager import ModelManager
    from dnnlab.data_loader import DataLoader
    import logging

    logger = logging.getLogger('dnnlab')

    logger.info('Program started')

    data_loader = DataLoader(args, dataframe_process, sep_cols)

    model_manager = ModelManager(args, data_loader, build_dnn_model,
                                 send_metric)

    model_manager.start()

    logger.info('Program ended')


def main():
    launch()


if __name__ == '__main__':
    main()
