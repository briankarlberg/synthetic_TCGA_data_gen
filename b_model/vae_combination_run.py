# blank run START, Data import and setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
import argparse
from pathlib import Path
import tensorflow as tf
from keras import backend as K
from keras.layers import (Input,  # want float.64 to go into this layer, two input layers (enc and dec)
                          Conv1D,
                          Dense,
                          Conv1DTranspose,
                          Flatten,
                          Lambda,
                          Reshape)
from keras.models import Model
from keras.losses import binary_crossentropy

print('libraries loaded')

date = datetime.today().strftime('%Y-%m-%d')
cohorts = ["BRCA"]
otlr_cuts = ["100k", "15k", "10k", "5k", "1k", "500"]
epochs = [15, 30, 45]
batch_size = [32, 64, 128]
version = '1d_model'

combinations = list(itertools.product(cohorts, otlr_cuts, epochs, batch_size))

# 1Dconv, do not rebuild in memory - build from fresh kernel only (blank run)


tf.compat.v1.disable_eager_execution()


def build_model(latent_dim: int, train_norm: np.array):
    def compute_latent(x):
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        eps = K.random_normal(shape=(batch, dim))
        return mu + K.exp(sigma / 2) * eps

    def kl_reconstruction_loss(true, pred):
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred))
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    tf.executing_eagerly()

    encoder_input = Input(shape=(train_norm.shape[1], 1,))

    encoder_conv = Conv1D(filters=8,
                          kernel_size=3,
                          activation='relu',
                          padding='same')(encoder_input)

    encoder_conv = Conv1D(filters=16,
                          kernel_size=3,
                          padding='same',
                          activation='relu')(encoder_conv)

    encoder = Flatten()(encoder_conv)

    mu = Dense(latent_dim)(encoder)
    sigma = Dense(latent_dim)(encoder)

    latent_space = Lambda(compute_latent, output_shape=(latent_dim,))([mu, sigma])

    conv_shape = K.int_shape(encoder_conv)

    # Decoder start
    decoder_input = Input(shape=(latent_dim,))

    decoder = Dense(conv_shape[1] * conv_shape[2], activation='relu')(decoder_input)
    decoder = Reshape((conv_shape[1], conv_shape[2]))(decoder)

    decoder_conv = Conv1DTranspose(filters=16,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu')(decoder)

    decoder_conv = Conv1DTranspose(filters=8,
                                   kernel_size=3,
                                   activation='relu',
                                   padding='same')(decoder_conv)

    decoder_conv = Conv1DTranspose(filters=1,
                                   kernel_size=3,
                                   activation='relu',
                                   padding='same')(decoder_conv)

    encoder = Model(encoder_input, latent_space)
    decoder = Model(decoder_input, decoder_conv)
    vae = Model(encoder_input, decoder(encoder(encoder_input)))
    vae.compile(optimizer='adam', loss=kl_reconstruction_loss)  # blank model set for (pre)training
    print('model built')
    return encoder, decoder, vae


def create_plots(cohort: str, epochs: int, otlr_cut: str, latent_dim: int, batch_size: int, date: str, version: str,
                 history):
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.title(cohort + ' embedding loss\n1D convolutional VAE'
              # ' pre-train\n'+
              )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.annotate('Outliers cut value = ' + otlr_cut +
                 '\nlatent dim = ' + str(latent_dim) +
                 '\nBatch size = ' + str(batch_size) +
                 '\nConvolutional layer count = 2\nTest ratio = .25\nNormalization = MinMax',
                 xy=(.4, .8), xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='top',
                 # fontsize=20
                 )

    plt.legend(loc="lower left")
    plt.savefig(Path(results_folder,
                     f'{cohort}_outlier_cut_{otlr_cut}_epochs_{str(epochs)}_latent_dim_{str(latent_dim)}_{date}_{version}.png'))
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", action="store", required=True,
                        help="combo index number", type=int)
    args = parser.parse_args()

    results_folder = Path("results")

    if not results_folder.exists():
        results_folder.mkdir(parents=True, exist_ok=True)

    selected_combination = combinations[args.index]

    cohort = selected_combination[0]
    otlr_cut = selected_combination[1]
    epochs = selected_combination[2]
    batch_size = selected_combination[3]
    latent_dim = 100

    # read pre-normalized data
    train_norm = pd.read_csv(
        '../a_data_structure/normalized_data/flat/' + cohort + '_X_train_flat_' + otlr_cut + '_otlr_cut_MinMax.tsv',
        sep='\t',
        index_col=0)

    test_norm = pd.read_csv(
        '../a_data_structure/normalized_data/flat/' + cohort + '_X_test_flat_' + otlr_cut + '_otlr_cut_MinMax.tsv',
        sep='\t',
        index_col=0)

    train_norm_arr = train_norm.to_numpy()
    train_norm_arr_exp = np.expand_dims(train_norm_arr, axis=-1)

    test_norm_arr = test_norm.to_numpy()
    test_norm_arr_exp = np.expand_dims(test_norm_arr, axis=-1)

    encoder, decoder, vae = build_model(latent_dim, train_norm_arr)

    # tf.keras.utils.plot_model(
    #     encoder,
    #     show_shapes=True,
    #     to_file=Path(results_folder, f'encoder_{cohort}_{otlr_cut}_{epochs}_{batch_size}_1D_model.png'))
    #
    # tf.keras.utils.plot_model(
    #     decoder,
    #     show_shapes=True,
    #     to_file=Path(results_folder, f"decoder_{cohort}_{otlr_cut}_{epochs}_{batch_size}_1D_model.png"))

    history = vae.fit(x=train_norm_arr_exp, y=train_norm_arr_exp, epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(test_norm_arr_exp, test_norm_arr_exp))

    create_plots(cohort=cohort, otlr_cut=otlr_cut, latent_dim=latent_dim, batch_size=batch_size, date=date,
                 version=version, history=history, epochs=epochs)

    trn_ltnt = encoder.predict(train_norm_arr_exp)
    tst_ltnt = encoder.predict(test_norm_arr_exp)
    trn_dec = decoder.predict(trn_ltnt)
    tst_dec = decoder.predict(tst_ltnt)

    trn_decDF = pd.DataFrame(np.squeeze(trn_dec))  # hoping the labels map from the raw file, do in UMAP
    tst_decDF = pd.DataFrame(np.squeeze(tst_dec))

    trn_decDF.to_csv(Path(results_folder, cohort + '_' +
                          # '_pretrain_'+
                          otlr_cut + '_outlier_cut_train_' +  # Train <------
                          str(epochs) + '_epochs_' +
                          str(latent_dim) + '_latent_dim_' +
                          date + '_' + version + '.tsv', sep='\t'))

    tst_decDF.to_csv(Path(results_folder, cohort + '_' +
                          # '_pretrain_'+
                          otlr_cut + '_outlier_cut_test_' +  # Test <------
                          str(epochs) + '_epochs_' +
                          str(latent_dim) + '_latent_dim_' +
                          date + '_' + version + '.tsv', sep='\t'))
