#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------- Importering af relevante biblioteker ----------------------#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold  # Changed from GroupKFold to KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import datetime
import mne
from mne.io import read_raw_brainvision
import os
from sklearn.model_selection import GroupKFold
import time
import pickle
import pandas as pd


#Import af datafiler
#EEG-filer
EEG_AA56D = read_raw_brainvision('/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AA56D/data/baseline_without_error/20230427_AA56D_orthosisErrorIjcai_multi_baseline_set1.vhdr', preload = True)
EEG_AC17D = read_raw_brainvision('/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AC17D/data/baseline_without_error/20230424_AC17D_orthosisErrorIjcai_multi_baseline_set2.vhdr', preload = True)
EEG_AJ05D = read_raw_brainvision('/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AJ05D/data/baseline_without_error/20230426_AJ05D_orthosisErrorIjcai_multi_baseline_set2.vhdr', preload = True)
EEG_AQ59D = read_raw_brainvision('/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AQ59D/data/baseline_without_error/20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1.vhdr', preload = True)
EEG_AW59D = read_raw_brainvision('/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AW59D/data/baseline_without_error/20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2.vhdr', preload = True)
EEG_AY63D = read_raw_brainvision('/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AY63D/data/baseline_without_error/20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2.vhdr', preload = True)
EEG_BS34D = read_raw_brainvision('/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/BS34D/data/baseline_without_error/20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1.vhdr', preload = True)
EEG_BY74D = read_raw_brainvision('/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/BY74D/data/baseline_without_error/20230424_BY74D_orthosisErrorIjcai_multi_baseline_set1.vhdr', preload = True)

EEG_DATA = [EEG_AJ05D, EEG_AQ59D, EEG_AW59D, EEG_AY63D, EEG_BS34D, EEG_BY74D]
SUBJECT_IDS = ['AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D']


#EMG-filer
EMG_DATA =[
'/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AJ05D/baseline_without_error/20230426_AJ05D_orthosisErrorIjcai_multi_baseline_set1.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AQ59D/baseline_without_error/20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AW59D/baseline_without_error/20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AY63D/baseline_without_error/20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/BS34D/baseline_without_error/20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/BY74D/baseline_without_error/20230424_BY74D_orthosisErrorIjcai_multi_baseline_set1.txt'
]


def plot_training_curves(history, fold_num, save_dir="training_curves_Transformer"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(history.history['accuracy'], label='Træning')
    axs[0].plot(history.history['val_accuracy'], label='Validering')
    axs[0].set_title(f'Fold {fold_num}: Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(history.history['loss'], label='Træning')
    axs[1].plot(history.history['val_loss'], label='Validering')
    axs[1].set_title(f'Fold {fold_num}: Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"fold_{fold_num}_curves.png"))
    plt.show()
    plt.close(fig)



#Transformer model
# ---------------------- Data forberedelse ----------------------

def prepare_data(eeg_raw, emg_raw, window_size=500, overlap=0.5, eeg_shift=200, scaler=None, apply_threshold=True, threshold=None):
    """
    Forbereder EEG og EMG data til træning af AI-modellen.
    Nu inkluderer scaler og threshold som parametre for at undgå data leakage.
    
    Parameters:
    -----------
    eeg_raw : Raw EEG data
    emg_raw : Raw EMG data
    window_size : Størrelse af hvert vindue i samples
    overlap : Overlap mellem vinduer (0-1)
    eeg_shift : EEG forskydning i ms
    scaler : Pre-fitted StandardScaler (hvis None, en ny laves)
    apply_threshold : Om der skal anvendes threshold til at finde bevægelser
    threshold : Pre-beregnet threshold (hvis None, beregnes)
    
    Returns:
    --------
    X : Vinduer af EEG data
    y : Tilsvarende bevægelses labels
    n_channels : Antal EEG kanaler
    scaler : Fitted StandardScaler
    threshold : Beregnet threshold
    """
    # Vælger hvilke EEG kanaler, der skal undersøges
    channels = ['C3', 'C4', 'CZ']
    available_channels = eeg_raw.ch_names
    channels = [ch for ch in channels if ch in available_channels]  # Sikrer kanalerne faktisk findes i datasættene
    if not channels:
        raise ValueError("Ingen valide kanaler fundet i EEG data.")

    # Får EEG samplingraten
    sfreq = eeg_raw.info['sfreq']

    # Gør intervallet 30s til 230s til samples
    start_sample = int(30 * sfreq)
    end_sample = int(230 * sfreq)

    # Her anvendes EEG forskydningen, for at have bevægelsesforberedelse in mente
    eeg_start = max(start_sample - int(eeg_shift * sfreq / 1000), 0)

    # Filtrering af EEG (0.5 - 45 Hz) og udtrækker kun data fra de valgte kanaler
    eeg_data = eeg_raw.copy().filter(0.5, 45).get_data(picks=channels)[:, eeg_start:end_sample]
    n_channels = len(channels)

    # Sikrer EMG er 1D
    if len(emg_raw.shape) > 1:
        emg_raw = emg_raw[:, 0]

    # Udtrækker det samme tidsvindue i EMG data, sikrer EEG og EMG matcher hinanden
    emg_data = emg_raw[start_sample:end_sample]

    # Normalisering af EEG - nu med mulighed for at genbruge scaler for at undgå data leakage
    if scaler is None:
        scaler = StandardScaler()
        eeg_data = scaler.fit_transform(eeg_data.T).T
    else:
        eeg_data = scaler.transform(eeg_data.T).T

    # Bevægelses-detektion baseret på EMG
    if apply_threshold:
        if threshold is None:
            # Laver en tærskel på T = Base + 1.2μ + 2σ, hvor μ er middelværdien, og σ er standardafvigelsen
            baseline = np.percentile(emg_data, 10)
            mean_emg = np.mean(emg_data)
            std_emg = np.std(emg_data)
            threshold = baseline + 1.2 * mean_emg + 2 * std_emg
        
        # Gør EMG til binære markeringer (1 = bevægelse, 0 = ingen bevægelse)
        emg_labels = (emg_data > threshold).astype(int)
    else:
        # Hvis vi bruger en pre-defineret threshold (f.eks. fra træningsdata)
        emg_labels = (emg_data > threshold).astype(int)

    # Et glidende vindue
    stride = int(window_size * (1 - overlap))
    n_windows = (len(emg_data) - window_size) // stride + 1

    # Forbereder datasæt, ved at oprette arrays til lagring af EEG (X) og bevægelsesmarkeringer (Y)
    X = np.zeros((n_windows, window_size, n_channels))
    y = np.zeros(n_windows)

    # Her udfyldes vinduerne --> EEG-seksvens (X) --> signalet for den valgte tidsperiode, og EMG-label (Y) --> den mest almindelige bevægelsesklasse i vinduet
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        X[i] = eeg_data[:, start_idx:end_idx].T
        y[i] = np.bincount(emg_labels[start_idx:end_idx]).argmax()

    return X, y, n_channels, scaler, threshold  # Returnerer også scaler og threshold

# ---------------------- Transformer Model ----------------------

def create_transformer_model(input_shape, num_classes=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)

    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_shape[0], 64)(positions)
    x = x + pos_embedding

    for _ in range(3):
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attention_output)
        ff = tf.keras.layers.Dense(64, activation='relu')(x)
        ff = tf.keras.layers.Dense(64)(ff)
        x = tf.keras.layers.LayerNormalization()(x + ff)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
# ---------------------- K-Fold Kryds Validation ----------------------

#Definiton af funktionen
def train_and_evaluate_kfold_transformer(eeg_list, emg_list, subject_ids, n_splits=5, window_size=500, epochs=50, batch_size=32):
    all_X, all_y, all_subject_ids = [], [], []
    for eeg, emg_path, subj in zip(eeg_list, emg_list, subject_ids):
        emg_raw = np.loadtxt(emg_path)
        X_subj, y_subj, _, scaler, threshold = prepare_data(eeg, emg_raw, window_size)
        all_X.append(X_subj)
        all_y.append(y_subj)
        all_subject_ids.extend([subj] * len(y_subj))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_ids_arr = np.array(all_subject_ids)

    fold_results, all_auc_scores, all_f1_scores = [], [], []
    all_inference_times, all_training_times = [], []
    kf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, groups=subject_ids_arr)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        if len(np.unique(y[test_idx])) < 2:
            print(f"⚠️ Fold {fold+1} ignoreres – kun én klasse i testdata.")
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = create_transformer_model(input_shape=X.shape[1:])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        start_train = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epochs, batch_size=batch_size, verbose=2)
        training_time = time.time() - start_train
        all_training_times.append(training_time)

        plot_training_curves(history, fold + 1, save_dir="training_curves_Transformer")

        start_inf = time.time()
        y_pred_proba = model.predict(X_test)
        inference_time = (time.time() - start_inf) / len(y_test)
        all_inference_times.append(inference_time)

        y_pred = np.argmax(y_pred_proba, axis=1)

        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        val_accuracy = max(history.history['val_accuracy'])

        fold_results.append(val_accuracy)
        all_auc_scores.append(auc_score)
        all_f1_scores.append(f1)

        print(f"✅ Fold {fold+1} | acc={val_accuracy:.4f}, AUC={auc_score:.4f}, F1={f1:.4f}")
        print(f"🕒 træning: {training_time:.2f}s | inferens: {inference_time:.6f}s/sample")

        model_dir = "saved_models_Transformer"
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, f"Transformer_fold{fold+1}.keras"))
        with open(os.path.join(model_dir, f"scaler_fold{fold+1}.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(model_dir, f"threshold_fold{fold+1}.pkl"), "wb") as f:
            pickle.dump(threshold, f)

    if not fold_results:
        print("❌ Ingen valide fold blev evalueret.")
        return None

    results_df = pd.DataFrame({
        "fold": list(range(1, len(fold_results)+1)),
        "val_accuracy": fold_results,
        "auc": all_auc_scores,
        "f1": all_f1_scores,
        "inference_time": all_inference_times,
        "training_time": all_training_times
    })

    avg = results_df.mean(numeric_only=True)
    avg["fold"] = "Avg"
    results_df = pd.concat([results_df, pd.DataFrame([avg])], ignore_index=True)

    csv_path = os.path.join(model_dir, "Transformer_kfold_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"📁 Evalueringsmetrikker gemt i: {csv_path}")
    return results_df

#---------------------- Påbegyndelse af træning ----------------------
if __name__ == "__main__":
    try:
        results = train_and_evaluate_kfold_transformer(
            EEG_DATA, EMG_DATA, SUBJECT_IDS,
            n_splits=5, window_size=500,
            epochs=50, batch_size=32
        )
        print(results)
    except Exception as e:
        print(f"Fejl: {str(e)}")
