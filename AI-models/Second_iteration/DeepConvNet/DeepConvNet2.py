#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 20:13:29 2025

@author: tobiasbendix
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 19:42:49 2025

@author: tobiasbendix
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import datetime
import mne
from mne.io import read_raw_brainvision
import time
import joblib
import os
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras import layers, models
model = models.Sequential()

def plot_training_curves(history, fold_num, save_dir="training_curves"):
    """
    Gemmer OG viser træningskurver (accuracy og loss) for hver fold.
    """
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
    plt.show()  # <--- viser grafen direkte
    plt.close(fig)

layers = tf.keras.layers
regularizers = tf.keras.regularizers

# Import af datafiler
# EEG-filer
EEG_AA56D = read_raw_brainvision(
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AA56D/data/baseline_without_error/20230427_AA56D_orthosisErrorIjcai_multi_baseline_set1.vhdr',
    preload=True)
EEG_AC17D = read_raw_brainvision(
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AC17D/data/baseline_without_error/20230424_AC17D_orthosisErrorIjcai_multi_baseline_set2.vhdr',
    preload=True)
EEG_AJ05D = read_raw_brainvision(
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AJ05D/data/baseline_without_error/20230426_AJ05D_orthosisErrorIjcai_multi_baseline_set2.vhdr',
    preload=True)
EEG_AQ59D = read_raw_brainvision(
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AQ59D/data/baseline_without_error/20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1.vhdr',
    preload=True)
EEG_AW59D = read_raw_brainvision(
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AW59D/data/baseline_without_error/20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2.vhdr',
    preload=True)
EEG_AY63D = read_raw_brainvision(
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/AY63D/data/baseline_without_error/20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2.vhdr',
    preload=True)
EEG_BS34D = read_raw_brainvision(
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/BS34D/data/baseline_without_error/20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1.vhdr',
    preload=True)
EEG_BY74D = read_raw_brainvision(
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EEG/BY74D/data/baseline_without_error/20230424_BY74D_orthosisErrorIjcai_multi_baseline_set1.vhdr',
    preload=True)

EEG_DATA = [EEG_AJ05D, EEG_AQ59D, EEG_AW59D, EEG_AY63D, EEG_BS34D, EEG_BY74D]
SUBJECT_IDS = ['AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D']

# EMG-filer
EMG_DATA = [
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AJ05D/baseline_without_error/20230426_AJ05D_orthosisErrorIjcai_multi_baseline_set1.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AQ59D/baseline_without_error/20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AW59D/baseline_without_error/20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AY63D/baseline_without_error/20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/BS34D/baseline_without_error/20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/BY74D/baseline_without_error/20230424_BY74D_orthosisErrorIjcai_multi_baseline_set1.txt'
]


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

def create_deepconvnet_model(input_shape, num_classes=2):
    """
    Optimeret DeepConvNet model til EEG-baseret bevægelsesdetektion
    med færre parametre og stærkere regularisering.
    """

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Reshape(input_shape)(inputs)

    # Block 1
    x = layers.Conv1D(filters=16, kernel_size=10, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=3)(x)

    # Block 2
    x = layers.Conv1D(filters=32, kernel_size=10, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=3)(x)

    # Block 3
    x = layers.Conv1D(filters=64, kernel_size=10, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=3)(x)

    # Block 4
    x = layers.Conv1D(filters=64, kernel_size=10, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=3)(x)

    # Output block
    x = layers.GlobalAveragePooling1D()(x)  # ↓ færre parametre end Flatten
    x = layers.Dropout(0.7)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def train_and_evaluate_kfold(eeg_list, emg_list, subject_ids, create_model_fn,
                             n_splits=5, window_size=500, epochs=50, batch_size=64):
    print("🧠 Forbereder hele datasættet...")
    all_X, all_y, all_subject_ids = [], [], []
    for eeg, emg_path, subj in zip(eeg_list, emg_list, subject_ids):
        print(f"📦 Forbereder data for {subj}...")
        emg_raw = np.loadtxt(emg_path)
        X_subj, y_subj, _, scaler, threshold = prepare_data(eeg_raw=eeg, emg_raw=emg_raw, window_size=window_size)
        all_X.append(X_subj)
        all_y.append(y_subj)
        all_subject_ids.extend([subj] * len(y_subj))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_ids_arr = np.array(all_subject_ids)

    print(f"🧠 Samlet dataset: X={X.shape}, y={y.shape}")
    print(f"👤 Unikke subjekter: {np.unique(subject_ids_arr)}")

    fold_results = []
    all_auc_scores = []
    all_f1_scores = []
    all_inference_times = []
    all_training_times = []

    kf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, groups=subject_ids_arr)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        test_subjects = np.unique(subject_ids_arr[test_idx])
        print(f"🧪 Test subjects: {test_subjects}")

        y_test_unique = np.unique(y[test_idx])
        print(f"📊 Test label distribution: {dict(zip(*np.unique(y[test_idx], return_counts=True)))}")

        if len(y_test_unique) < 2:
            print(f"⚠️ Fold {fold+1} ignoreres – kun én klasse i testdata.")
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = create_model_fn(input_shape=X.shape[1:])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Træning
        start_train = time.time()
        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=2)
        end_train = time.time()
        training_time = end_train - start_train
        all_training_times.append(training_time)

        plot_training_curves(history, fold + 1)

        # Inferens
        start_inf = time.time()
        y_pred_proba = model.predict(X_test)
        end_inf = time.time()
        inference_time = (end_inf - start_inf) / len(y_test)
        all_inference_times.append(inference_time)

        y_pred = np.argmax(y_pred_proba, axis=1)

        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        val_accuracy = max(history.history['val_accuracy'])

        fold_results.append(val_accuracy)
        all_auc_scores.append(auc_score)
        all_f1_scores.append(f1)

        print(f"✅ Fold {fold+1} | acc={val_accuracy:.4f}, AUC={auc_score:.4f}, F1={f1:.4f}, prec={precision:.4f}, rec={recall:.4f}")
        print(f"🕒 træning: {training_time:.2f}s | inferens: {inference_time:.6f}s/sample")

        # Gem model og preprocessing artefakter
        model_path = f"saved_models_DeepConvNet/deepconvnet_fold{fold+1}.keras"
        model.save(model_path)
        print(f"💾 Model gemt: {model_path}")

        with open(f"saved_models_DeepConvNet/scaler_fold{fold+1}.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(f"saved_models_DeepConvNet/threshold_fold{fold+1}.pkl", "wb") as f:
            pickle.dump(threshold, f)

    if not fold_results:
        print("❌ Ingen valide fold blev evalueret.")
        return None

    # Samlet resultattabel
    results_df = pd.DataFrame({
        "fold": list(range(1, len(fold_results)+1)),
        "val_accuracy": fold_results,
        "auc": all_auc_scores,
        "f1": all_f1_scores,
        "inference_time": all_inference_times,
        "training_time": all_training_times
    })

    # Gennemsnit
    averages = results_df.mean(numeric_only=True)
    averages["fold"] = "Avg"
    results_df = pd.concat([results_df, pd.DataFrame([averages])], ignore_index=True)

    # Gem og vis
    os.makedirs("saved_models_DeepConvNet", exist_ok=True)
    csv_path = "saved_models_DeepConvNet/deepconvnet_kfold_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"📁 Evalueringsmetrikker gemt i: {csv_path}")

    return results_df
# ---------------------- Kør træning ----------------------#
if __name__ == "__main__":
    try:
        print("Starting K-fold cross-validation training...")
        results_df = train_and_evaluate_kfold(
            eeg_list=EEG_DATA, 
            emg_list=EMG_DATA,
            subject_ids=SUBJECT_IDS,
            create_model_fn=create_deepconvnet_model,
            n_splits=5,
            window_size=500,
            epochs=50,
            batch_size=32
        )

        print("Training and evaluation completed successfully.")
        avg_metrics = results_df[results_df["fold"] == "Avg"]
        print("\n--- Gennemsnitlige evalueringsmetrikker ---")
        print(f"Accuracy:       {avg_metrics['val_accuracy'].values[0]:.4f}")
        print(f"AUC:            {avg_metrics['auc'].values[0]:.4f}")
        print(f"F1-score:       {avg_metrics['f1'].values[0]:.4f}")
        print(f"Inferenstid:    {avg_metrics['inference_time'].values[0]:.6f} sek/sample")
        print(f"Træningstid:    {avg_metrics['training_time'].values[0]:.2f} sek/fold")

    except Exception as e:
        print(f"Error: {str(e)}")
