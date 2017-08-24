def build_dnn_model(model_name, max_feature, input_length):
    """Build and compile Keras models.

    # Example

    ```python
        from keras.models import Sequential, Model
        from keras.layers import Dense, Embedding, Flatten
        # multilayer perceptrons
        if model_name == 'mlp':
            model = Sequential()
            model.add(Dense(128, activation='relu', input_length=input_length))
            model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='adagrad',
            loss='binary_crossentropy',
            metrics=['binary_crossentropy'])
        return model
    ```

    # Arguments
        model_name: name of the model to use (you can build serveral models
          and use them by giving them a name in the 'if' condition).
        max_feature: only relevant when using 'Embedding' layer.
        input_length: length of input sequences.

    # Return
        Compiled Keras model

    """

    from keras.models import Sequential
    from keras.layers import Dense
    # multilayer perceptrons
    if model_name == 'mlp':
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(input_length, )))
        model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adagrad',
        loss='binary_crossentropy',
        metrics=['binary_crossentropy'])

    return model


from dnnlab import main_dnnlab

main_dnnlab.launch(args=None,
    build_dnn_model=build_dnn_model)
