#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------- Importering af relevante biblioteker ----------------------#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import datetime
import joblib
from mne.io import read_raw_brainvision
import mne
import time  # Added for timing functionality

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

#EMG-filer
EMG_DATA =[
'/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AJ05D/baseline_without_error/20230426_AJ05D_orthosisErrorIjcai_multi_baseline_set1.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AQ59D/baseline_without_error/20230421_AQ59D_orthosisErrorIjcai_multi_baseline_set1.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AW59D/baseline_without_error/20230425_AW59D_orthosisErrorIjcai_multi_baseline_set2.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/AY63D/baseline_without_error/20230425_AY63D_orthosisErrorIjcai_multi_baseline_set2.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/BS34D/baseline_without_error/20230426_BS34D_orthosisErrorIjcai_multi_baseline_set1.txt',
    '/Users/tobiasbendix/Desktop/UngeForskere25(theLastDance)/EMGogEEG(orthosis)/EMG/BY74D/baseline_without_error/20230424_BY74D_orthosisErrorIjcai_multi_baseline_set1.txt'
]
#Transformer model
# ---------------------- Data forberedelse ----------------------

def prepare_data(eeg_raw, emg_raw, window_size=500, overlap=0.1, eeg_shift=200): #Definition af funktion der forbereder EEG og EMG data til træning af AI-modellen, hvor "eeg_shift" er antallet af ms signalet forskydes

    #Vælger hvilke EEG kanaler, der skal undersøges
    channels = ['C3', 'C4', 'CZ']
    available_channels = eeg_raw.ch_names
    channels = [ch for ch in channels if ch in available_channels]              #Sikrer kanalerne faktisk findes i datasættene
    if not channels:
        raise ValueError("Ingen valide kanaler fundet i EEG data.")

    #Får EEG samplingraten
    sfreq = eeg_raw.info['sfreq']

    #Gør intervallet 30s til 230s til samples
    start_sample = int(30 * sfreq)
    end_sample = int(230 * sfreq)

    #Her anvendes EEG forskydningen, for at have bevægelsesforberedelse in mente
    eeg_start = max(start_sample - int(eeg_shift * sfreq / 1000), 0)

    # Filtrering af EEG (0.5 - 45 Hz) og udtrækker kun data fra de valgte kanaler
    eeg_data = eeg_raw.copy().filter(0.5, 45).get_data(picks=channels)[:, eeg_start:end_sample]
    n_channels = len(channels)

    #Sikrer EMG er 1D
    if len(emg_raw.shape) > 1:
        emg_raw = emg_raw[:, 0]

    #Udtrækker det samme tidsvindue i EMG data, sikrer EEG og EMG matcher hinanden
    emg_data = emg_raw[start_sample:end_sample]

    #Normalisering af EEG
    scaler_eeg = StandardScaler()
    eeg_data = scaler_eeg.fit_transform(eeg_data.T).T

    #Laver en tærskel på T = Base + 1.2μ + 2σ, hvor μ er middelværdien, og σ er standardafvigelsen
    baseline = np.percentile(emg_data, 10)
    mean_emg = np.mean(emg_data)
    std_emg = np.std(emg_data)
    threshold = baseline + 1.2 * mean_emg + 2 * std_emg

    #Gør EMG til binære markeringer (1 = bevægelse, 0 = ingen bevægelse)
    emg_labels = (emg_data > threshold).astype(int)

    #Et glidende vindue
    stride = int(window_size * (1 - overlap))
    n_windows = (len(emg_data) - window_size) // stride + 1

    print(f"Laver {n_windows} vinduer...") #statusbesked til debugging

    #Forbereder datasæt, ved at oprette arrays til lagring af EEG (X) og bevægelsesmarkeringer (Y)
    X = np.zeros((n_windows, window_size, n_channels))
    y = np.zeros(n_windows)

    #Her udfyldes vinduerne --> EEG-seksvens (X) --> signalet for den valgte tidsperiode, og EMG-label (Y) --> den mest almindelige bevægelsesklasse i vinduet
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        X[i] = eeg_data[:, start_idx:end_idx].T
        y[i] = np.bincount(emg_labels[start_idx:end_idx]).argmax()

    return X, y, n_channels  # Returnerer de færdigbehandlede data

# ---------------------- Transformer Model ----------------------

def create_transformer_model(input_shape, num_classes=2): #Definerer Transformer-modellen

    #inputlaget, som tager EEG-data (vinduesstørrelser, kanaler)
    inputs = tf.keras.layers.Input(shape=input_shape)

    #Udvider feature-space fra EEG-signalet til 128 features
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)

    # Positional Encoding --> på den måde forstår modellen tidsrækkefølgen
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_shape[0], 128)(positions)
    x = x + pos_embedding

    #De 3 transformer blokke: 1 -> Multihead-Attention, 2 -> Residual Connection, 3 -> Feedforward-netværk
    for _ in range(3):
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = tf.keras.layers.LayerNormalization()(attention_output + x)
        ff = tf.keras.layers.Dense(128, activation='relu')(x)
        ff = tf.keras.layers.Dense(128)(ff)
        x = tf.keras.layers.LayerNormalization()(ff + x)

    #samler de vigtigste informationer fra sekvensen
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layers, som reducerer over-fitting
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    #Softmax-output: Klassificere ml. bevægelse (1) og ingen bevægelse (0)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    #Retunerer de færdige transformer-model
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# ---------------------- K-Fold Kryds Validation ----------------------

#Definiton af funktionen
def train_and_evaluate(eeg_list, emg_list, window_size=500, epochs=100, batch_size=32, n_splits=5):
    print("Påbegynder træningspipeline med K-fold Cross-Validation")
    
    # Start timer for overall training process
    overall_start_time = time.time()

    X_all = []
    y_all = []
    num_channels = None

    # Timing data preparation
    data_prep_start_time = time.time()
    for eeg_raw, emg_path in zip(eeg_list, emg_list):
        print(f"Processing {emg_path}...")
        emg_data = np.loadtxt(emg_path)
        X, y, n_channels = prepare_data(eeg_raw, emg_data, window_size=window_size)

        X_all.append(X)
        y_all.append(y)
        num_channels = n_channels

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    data_prep_time = time.time() - data_prep_start_time
    print(f"Data forberedelse afsluttet på {data_prep_time:.2f} sekunder")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_confusion_matrices = []
    all_auc_scores = []
    all_fpr = []
    all_tpr = []
    fold_times = []  # List to store timing for each fold

    plt.figure(figsize=(15, 10))  # Increased figure size to accommodate timing info

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        fold_start_time = time.time()  # Start timer for this fold
        print(f"\nTræningsfold {fold + 1}/{n_splits}")

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        model = create_transformer_model(input_shape=(window_size, num_channels))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(f'model_fold{fold}_{timestamp}.keras',
                                               save_best_only=True,
                                               monitor='val_loss'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             restore_best_weights=True)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        # Get predictions and calculate metrics
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        all_confusion_matrices.append(cm)

        # ROC Curve and AUC Score
        fpr, tpr, _ = roc_curve(y_val, y_pred[:, 1])
        auc_score = auc(fpr, tpr)
        all_auc_scores.append(auc_score)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

        val_accuracy = max(history.history['val_accuracy'])
        fold_results.append(val_accuracy)
        
        # Calculate and store the time taken for this fold
        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)
        print(f"Fold {fold + 1} afsluttet på {fold_time:.2f} sekunder")

        # Plot ROC curve for this fold
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, label=f'Fold {fold + 1} (AUC = {auc_score:.2f})')

    # Calculate and plot average confusion matrix
    avg_cm = np.mean(all_confusion_matrices, axis=0)
    plt.subplot(2, 2, 2)
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Average Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Finalize ROC curve plot
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Folds')
    plt.legend(loc="lower right")

    # Plot metrics summary with additional timing info
    plt.subplot(2, 2, 3)
    plt.axis('off')
    
    # Calculate overall training time
    overall_training_time = time.time() - overall_start_time
    
    summary_text = f'Performance Metrics:\n\n' \
                   f'Average Accuracy: {np.mean(fold_results):.4f}\n' \
                   f'Average AUC: {np.mean(all_auc_scores):.4f}\n' \
                   f'Std Dev AUC: {np.std(all_auc_scores):.4f}'
    plt.text(0.1, 0.5, summary_text, fontsize=12)
    
    # Plot timing information
    plt.subplot(2, 2, 4)
    plt.axis('off')
    timing_text = f'Timing Information:\n\n' \
                  f'Data Preparation: {data_prep_time:.2f} seconds\n' \
                  f'Overall Training Time: {overall_training_time:.2f} seconds\n\n' \
                  f'Fold Timing (seconds):\n'
                  
    for i, time_taken in enumerate(fold_times):
        timing_text += f'Fold {i+1}: {time_taken:.2f}\n'
        
    timing_text += f'\nAverage Fold Time: {np.mean(fold_times):.2f} seconds'
    plt.text(0.1, 0.5, timing_text, fontsize=12)

    plt.tight_layout()
    plt.savefig(f'performance_metrics_{timestamp}.png')
    plt.close()

    avg_accuracy = np.mean(fold_results)
    avg_auc = np.mean(all_auc_scores)

    print(f"\nGennemsnitlig validation præcision over alle folder: {avg_accuracy:.4f}")
    print(f"Gennemsnitlig AUC score over alle folder: {avg_auc:.4f}")
    print(f"\nConfusion Matrix (gennemsnit over alle folder):")
    print(avg_cm)
    
    # Print timing summary
    print("\nTiming Information:")
    print(f"Data Preparation: {data_prep_time:.2f} seconds")
    print(f"Overall Training Time: {overall_training_time:.2f} seconds")
    print("\nFold Timing (seconds):")
    for i, time_taken in enumerate(fold_times):
        print(f"Fold {i+1}: {time_taken:.2f}")
    print(f"\nAverage Fold Time: {np.mean(fold_times):.2f} seconds")

    plt.show()
    return avg_accuracy, avg_auc, avg_cm, fold_times, overall_training_time

#---------------------- Påbegyndelse af træning ----------------------
train_and_evaluate(eeg_list=EEG_DATA, emg_list=EMG_DATA, window_size=500, epochs=100, batch_size=32, n_splits=5)