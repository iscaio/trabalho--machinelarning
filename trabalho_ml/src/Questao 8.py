import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Surprise imports
from surprise import Dataset, Reader, KNNBasic, SVD, NMF
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# TensorFlow for autoencoder
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def load_movielens_ml100k(path="u.data"):
    df = pd.read_csv(path, sep='\t', names=['userId','movieId','rating','timestamp'], engine='python')
    return df

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def surprise_user_item_and_svd(df):
    reader = Reader(rating_scale=(df.rating.min(), df.rating.max()))
    data = Dataset.load_from_df(df[['userId','movieId','rating']], reader)
    trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

    results = {}

    sim_options = {'name': 'pearson', 'user_based': True}
    algo_user = KNNBasic(sim_options=sim_options, verbose=False, k=40)
    algo_user.fit(trainset)
    pred_user = algo_user.test(testset)
    rmse_user = accuracy.rmse(pred_user, verbose=False)
    mae_user = accuracy.mae(pred_user, verbose=False)
    results['user_knn'] = {'rmse': rmse_user, 'mae': mae_user}

    sim_options = {'name': 'pearson', 'user_based': False}
    algo_item = KNNBasic(sim_options=sim_options, verbose=False, k=40)
    algo_item.fit(trainset)
    pred_item = algo_item.test(testset)
    rmse_item = accuracy.rmse(pred_item, verbose=False)
    mae_item = accuracy.mae(pred_item, verbose=False)
    results['item_knn'] = {'rmse': rmse_item, 'mae': mae_item}

    algo_svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    algo_svd.fit(trainset)
    pred_svd = algo_svd.test(testset)
    rmse_svd = accuracy.rmse(pred_svd, verbose=False)
    mae_svd = accuracy.mae(pred_svd, verbose=False)
    results['svd'] = {'rmse': rmse_svd, 'mae': mae_svd}

    algo_nmf = NMF(n_factors=50, n_epochs=50, random_state=42)
    algo_nmf.fit(trainset)
    pred_nmf = algo_nmf.test(testset)
    results['nmf'] = {'rmse': accuracy.rmse(pred_nmf, verbose=False), 'mae': accuracy.mae(pred_nmf, verbose=False)}

    return results

def build_and_train_autoencoder(df, latent_dim=64, epochs=20, batch_size=128):
    users = df.userId.unique()
    items = df.movieId.unique()
    user_to_idx = {u:i for i,u in enumerate(sorted(users))}
    item_to_idx = {m:i for i,m in enumerate(sorted(items))}

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    R = np.zeros((n_users, n_items), dtype=np.float32)
    mask = np.zeros_like(R, dtype=np.float32)
    for _, row in df.iterrows():
        u = user_to_idx[row.userId]
        m = item_to_idx[row.movieId]
        R[u,m] = row.rating
        mask[u,m] = 1.0

    rng = np.random.RandomState(42)
    train_mask = mask.copy()
    test_pairs = []
    for u in range(n_users):
        rated = np.where(mask[u]==1)[0]
        if len(rated) < 2:
            continue
        test_idx = rng.choice(rated, size=max(1, int(0.2 * len(rated))), replace=False)
        for t in test_idx:
            train_mask[u,t] = 0
            test_pairs.append((u,t,R[u,t]))

    R_train = R * train_mask

    inp = layers.Input(shape=(n_items,))
    x = layers.GaussianNoise(0.1)(inp)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(latent_dim, activation='relu', activity_regularizer=regularizers.l2(1e-6))(x)
    x = layers.Dense(512, activation='relu')(x)
    out = layers.Dense(n_items, activation='linear')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')

    optimizer = tf.keras.optimizers.Adam()
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    R_train_tf = tf.convert_to_tensor(R_train, dtype=tf.float32)
    mask_train_tf = tf.convert_to_tensor(train_mask, dtype=tf.float32)

    steps_per_epoch = max(1, n_users // batch_size)

    for epoch in range(epochs):
        idx = np.arange(n_users)
        rng.shuffle(idx)
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch_idx = idx[step*batch_size:(step+1)*batch_size]
            X_batch = tf.gather(R_train_tf, batch_idx)
            M_batch = tf.gather(mask_train_tf, batch_idx)
            with tf.GradientTape() as tape:
                pred = model(X_batch, training=True)
                loss_matrix = mse_loss(X_batch * M_batch, pred * M_batch)
                denom = tf.reduce_sum(M_batch) + 1e-8
                loss = tf.reduce_sum(loss_matrix * M_batch) / denom
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_loss += loss.numpy()
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/steps_per_epoch:.6f}")

    preds = []
    truths = []
    recon = model.predict(R_train, batch_size=256)
    for (u,i,true_r) in test_pairs:
        pred_r = recon[u,i]
        preds.append(pred_r)
        truths.append(true_r)

    return {'preds': np.array(preds), 'truths': np.array(truths)}

def main():
    path = "u.data"
    if not os.path.exists(path):
        print("Coloque o arquivo 'u.data' (MovieLens 100k) no mesmo diretório ou atualize o path.")
        return

    df = load_movielens_ml100k(path)
    print("Dataset carregado:", df.shape)

    print("Executando métodos do Surprise (KNN, SVD, NMF)...")
    results = surprise_user_item_and_svd(df)
    print("Resultados (Surprise):")
    for k,v in results.items():
        print(f"  {k}: RMSE={v['rmse']:.4f}, MAE={v['mae']:.4f}")

    print("Treinando Autoencoder...")
    ae_res = build_and_train_autoencoder(df)
    preds = ae_res['preds']
    truths = ae_res['truths']
    print("Autoencoder: RMSE:", rmse(truths, preds), "MAE:", mae(truths, preds))

if __name__ == '__main__':
    main()
