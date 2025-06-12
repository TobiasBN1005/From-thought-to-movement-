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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import GroupKFold
import datetime
import mne
from mne.io import read_raw_brainvision
import time

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
    Creates a DeepConvNet model for EEG-based movement detection.
    This model architecture is based on the paper "Deep learning with convolutional
    neural networks for EEG decoding and visualization" (Schirrmeister et al., 2017).
    
    Parameters:
    -----------
    input_shape : Tuple with format (window_size, num_channels)
    num_classes : Number of classes (default: 2 - movement/no movement)
    
    Returns:
    --------
    model : Compiled Keras model
    """
    #Input lag som tager EEG-data (window_size, channels)
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Reshape, for at sikre dimensioner er rigtige for Conv1D
    x = tf.keras.layers.Reshape(input_shape)(inputs)
    
    #Første convolutional blok - temporal filtrering 
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=3)(x)
    
    #Andet convolutional block
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=10, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=3)(x)
    
    #Tredej convolutional block
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=10, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=3)(x)
    
    #Fjerne convolutional block
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=10, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=3)(x)
    
    #Klassfikationsblock
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output lag med softmax aktivation
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    #Laver og retunerer modellen 
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Print model resume, for at bekræfte arkitekturen
    model.summary()
    
    return model


def train_and_evaluate_kfold(eeg_list, emg_list, subject_ids, n_splits=5, window_size=500, epochs=100, batch_size=32):
    """
    Trains and evaluates the model using K-fold cross-validation.
    Tracks and reports execution time.
    
    Parameters:
    -----------
    eeg_list : List of EEG data for each subject
    emg_list : List of EMG file paths for each subject
    subject_ids : List of subject identifiers
    n_splits : Number of folds for K-fold cross validation
    window_size : Size of each window in samples
    epochs : Number of epochs for training
    batch_size : Batch size for training
    
    Returns:
    --------
    avg_accuracy : Average accuracy across all folds
    avg_auc : Average AUC score across all folds
    """
    
    #Start den overordnede timer
    overall_start_time = time.time()
    
    print(f"Starting {n_splits}-fold cross-validation training and evaluation...")

    all_data = []
    
    #Forbereder al data
    for i, (eeg_raw, emg_path, subject_id) in enumerate(zip(eeg_list, emg_list, subject_ids)):
        print(f"Preparing data for {subject_id}...")
        try:
            emg_data = np.loadtxt(emg_path)
            X, y, n_channels, _, _ = prepare_data(eeg_raw, emg_data, window_size=window_size)
            
            if X.shape[0] == 0:
                print(f"Warning: No data for subject {subject_id}, skipping.")
                continue
                
            all_data.append((X.astype(np.float32), y.astype(np.int32), subject_id))
            
        except Exception as e:
            print(f"Error processing {subject_id}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No valid data found for any subject.")

    # Concatenate al data
    X_all = np.concatenate([data[0] for data in all_data], axis=0)
    y_all = np.concatenate([data[1] for data in all_data], axis=0)
    
    num_channels = X_all.shape[2]
    print(f"Number of channels detected: {num_channels}")
    print(f"Total dataset shape - X: {X_all.shape}, y: {y_all.shape}")
    
    fold_results = []
    all_confusion_matrices = []
    all_auc_scores = []
    fold_training_times = []  # Gemmer tiden taget pr. fold
    
    # Laver K-fold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_all)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        
        # Start timing this fold
        fold_start_time = time.time()

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Laver og kompilerer model
        model = create_deepconvnet_model(input_shape=(window_size, num_channels))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(f'cnn_model_fold{fold+1}_{timestamp}.keras', save_best_only=True, monitor='val_loss'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]

        #Begynder træning
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
        except Exception as e:
            print(f"Error during training: {str(e)}")
            continue

        # Stoper tiden for denne fold
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_training_times.append(fold_duration)
        
        # Printer træningstiden for denne fold f
        minutes, seconds = divmod(fold_duration, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"\nFold {fold + 1} training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

        #forudsigelse og evaluering
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        all_confusion_matrices.append(cm)

        # ROC Curve and AUC Score
        if y_pred.shape[1] > 1:
            class1_preds = y_pred[:, 1]
        else:
            class1_preds = y_pred.flatten()
            
        fpr, tpr, _ = roc_curve(y_test, class1_preds)
        auc_score = auc(fpr, tpr)
        all_auc_scores.append(auc_score)

        val_accuracy = max(history.history['val_accuracy'])
        fold_results.append(val_accuracy)

        print(f"Fold {fold + 1} test accuracy: {val_accuracy:.4f}, AUC: {auc_score:.4f}")
        print("Confusion matrix:")
        print(cm)
        
        #Laver klassifikations report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))

    if not all_confusion_matrices:
        raise ValueError("No folds were completed. Check data and model.")

    avg_accuracy = np.mean(fold_results)
    avg_auc = np.mean(all_auc_scores)
    std_accuracy = np.std(fold_results)

    # Stoper den overrende tid
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    
    # Print total trænningstid
    minutes, seconds = divmod(overall_duration, 60)
    hours, minutes = divmod(minutes, 60)
    
    print("\n" + "="*50)
    print(f"K-FOLD CROSS-VALIDATION RESULTS ({n_splits} FOLDS)")
    print("="*50)
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")
    print("\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Print tid per fold
    for i, fold_time in enumerate(fold_training_times):
        minutes, seconds = divmod(fold_time, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"Fold {i+1} training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Gennemsnitlig fold tid
    avg_fold_time = np.mean(fold_training_times)
    minutes, seconds = divmod(avg_fold_time, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Average fold training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    return avg_accuracy, avg_auc

# ---------------------- Kør træning ----------------------#
if __name__ == "__main__":
    try:
        print("Starting K-fold cross-validation training...")
        avg_accuracy, avg_auc = train_and_evaluate_kfold(
            eeg_list=EEG_DATA, 
            emg_list=EMG_DATA,
            subject_ids=SUBJECT_IDS,
            n_splits=5,  # Use 5-fold cross-validation
            window_size=500,
            epochs=50,
            batch_size=32
        )
        print("Training and evaluation completed successfully.")
        print(f"Final average accuracy: {avg_accuracy:.4f}")
        print(f"Final average AUC: {avg_auc:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")