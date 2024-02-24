import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

import optuna
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState

import libdnnmcmc.se_NN_lib as NN
from libdnnmcmc.utility import import_training_data, load_scenario


testcase='loop'      # either 'loop' or 'tree'
trainset_path = "data_files/results_loop/MC_results_pt_1.csv"

SE, _, d_prior_dist, data_file = load_scenario(testcase)

# Training Config
n_train = 50000
n_epochs = 10  
batch_size = 32 
NN_type = "CNN"

def load_data(trainset_path, n_train)->tuple:
    """
    Read training data from csv file and split into training and validation sets
    trainset_path: str, path to csv file containing training data
    n_train: int, number of training samples
    """

    df = pd.read_csv(trainset_path, header=None)

    # Split into 80% for training, 20% for validation
    n_train_val = int(n_train / 0.8)    
    x_data, y_data = import_training_data(data_file, n_train_val)
    x_train = x_data[:n_train, :]
    y_train = y_data[:n_train, :]
    x_val = x_data[n_train:, :]
    y_val = y_data[n_train:, :]

    return x_train, y_train, x_val, y_val

def create_model(trial, NN_type="DNN"):
    """
    Optimized hyperparameters:
        DNN:
            n_layers
            hidden units
            dropout rate in each layer
            Optimizer
            learning rate of Optimizer
    """
    # 1.Config
    mask_matrix = SE.mask_matrix_full
    n_demands = SE.n_demands
    n_nodes = SE.n_nodes
    n_edges = SE.n_edges

    # linear scaling parameters for inputs - scale to 0-1
    x_train, _, _, _ = load_data(trainset_path, n_train)
    input_scale = tf.constant(1 / np.max(x_train, axis=0), dtype=tf.float64)
    input_offset = tf.zeros_like(input_scale)
    layer_input_scale = NN.MyScalingLayer(offset=tf.expand_dims(input_offset, axis=-1),
                                        scaling=input_scale,
                                        # mapping_matrix=tf.sparse.eye(n_demands),
                                        name='downscaled')

    # linear scaling parameters outputs - scale to zero mean ~ unit variance based on linearisation model
    prior_state_mean = tf.concat([SE.T, SE.mf, SE.p, SE.T_end], axis=0)
    jac = SE.evaluate_state_equations('demand jacobian')
    prior_state_cov = jac @ d_prior_dist.covariance() @ tf.transpose(jac)
    output_scale = tf.math.sqrt(tf.math.sqrt(tf.linalg.diag_part(prior_state_cov)))
    layer_output_scale = NN.MyScalingLayer(offset=prior_state_mean,
                                        scaling=output_scale,
                                        mapping_matrix=SE.mask_matrix_full,
                                        name='upscaled')

    n_latent_outputs = tf.shape(mask_matrix)[1]
    n_timesteps = 1   # For RNN

    # 2.Construct the model
    
    if NN_type == "DNN":
        model = keras.Sequential()
        n_layers = trial.suggest_int("n_layers", 1, 3)
        model.add(keras.Input(shape=(n_demands,)))
        model.add(layer_input_scale)
        for i in range(n_layers):
            num_hidden = trial.suggest_int("n_units_l{}".format(i), 32, 512, log=True)
            model.add(layers.Dense(num_hidden, activation='relu'))
            dropout_rate = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))
        model.add(layers.Dense(n_latent_outputs, activation='linear', name='unscaled'))
        model.add(layer_output_scale)
    
    elif NN_type == "CNN":
        conv1d_neurons0 = trial.suggest_int("conv1d_neurons", 32, 512, log=True)
        conv1d_neurons1 = trial.suggest_int("conv1d_neurons", 32, 512, log=True)
        dense_neurons = trial.suggest_int("dense_neurons", 32, 512, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)

        inputs = keras.Input(shape=(n_demands, n_timesteps))  # (4,1)
        x = layers.Conv1D(conv1d_neurons0, kernel_size=2, activation='relu')(inputs)
        x = layers.MaxPooling1D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(conv1d_neurons1, kernel_size=1, activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(dense_neurons, activation='relu')(x)
        x = layers.Dense(n_latent_outputs, activation='linear', name='unscaled')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        pred_states = layer_output_scale(x)
        model = keras.Model(inputs=inputs, outputs=[pred_states], name='1DCNN_model')

    elif NN_type == "LSTM":
        lstm_neurons0 = trial.suggest_int("lstm_neurons", 32, 512, log=True)
        lstm_neurons1 = trial.suggest_int("lstm_neurons", 32, 512, log=True)
        dense_neurons = trial.suggest_int("dense_neurons", 32, 512, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)

        inputs = keras.Input(shape=(n_demands, n_timesteps))
        x = layers.LSTM(lstm_neurons0, return_sequences=True)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(lstm_neurons1, return_sequences=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(dense_neurons, activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(n_latent_outputs, activation='linear', name='unscaled')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        pred_states = layer_output_scale(x)
        model = keras.Model(inputs=inputs, outputs=[pred_states], name='LSTM_model')
        model.summary()    
    
    elif NN_type == "BiLSTM":
        lstm_neurons0 = trial.suggest_int("lstm_neurons", 32, 512, log=True)
        lstm_neurons1 = trial.suggest_int("lstm_neurons", 32, 512, log=True)
        dense_neurons = trial.suggest_int("dense_neurons", 32, 512, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)

        inputs = keras.Input(shape=(n_demands, n_timesteps))
        x = layers.Bidirectional(layers.LSTM(lstm_neurons0, return_sequences=True))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Bidirectional(layers.LSTM(lstm_neurons1, return_sequences=False))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(dense_neurons, activation='tanh')(x)
        x = layers.Dense(n_latent_outputs, activation='linear', name='unscaled')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        pred_states = layer_output_scale(x)
        model = keras.Model(inputs=inputs, outputs=[pred_states], name='BiLSTM_model')
    
    model.summary()

    # 3.Compile the model
    L_WMSE = NN.LossWeightedMSE(n_nodes, n_edges, lambda_T=1, lambda_mf=5.e2, lambda_p=1, lambda_Tend=1)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSProp"])
    if optimizer == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "RMSProp":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    model.compile(optimizer=optimizer, 
                  loss=L_WMSE, 
                  metrics={'upscaled': [keras.metrics.MeanAbsolutePercentageError(),
                            NN.MetricMAPE_T(n_nodes, n_edges),
                            NN.MetricMAPE_mf(n_nodes, n_edges),
                            NN.MetricMAPE_p(n_nodes, n_edges),
                            NN.MetricMAPE_Tend(n_nodes, n_edges)
                            ]})
    return model

def objective(trial):
    keras.backend.clear_session()

    # 1.Load data
    x_train, y_train, x_val, y_val = load_data(trainset_path, n_train)
    
    if NN_type in ["LSTM", "BiLSTM"]:
        # reshape x to (4,1,1)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    # 2.Get a model
    model = create_model(trial, NN_type=NN_type)

    # 3.Train the model
    model.fit(
        x_train, 
        y_train,
        batch_size=batch_size,
        callbacks=[KerasPruningCallback(trial, "val_loss")],
        epochs=n_epochs,
        validation_data=(x_val, y_val),
        verbose=1    # print progress
    )

    # 4.Evaluate the model
    score = model.evaluate(x_val, y_val, verbose=1)
    return score[0]   # val loss

if __name__ == "__main__":
    n_trials = 500

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    # Results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))
    print("Best trial: ")
    trial = study.best_trial
    print("Value: ", trial.value)
    print("Params: ")
    for k, v in trial.params.items():
        print("{}: {}".format(k, v))

    # Write to file
    df = study.trials_dataframe()
    df.to_csv(f"data_files/results_loop/{NN_type}_optuna_results.csv")