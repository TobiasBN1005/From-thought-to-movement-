#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:10:38 2025

@author: tobiasbendix
Modified with K-fold cross-validation and timing metrics
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
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

def create_lstm_rnn_model(input_shape, num_classes=2):
    """
    Opretter en LSTM-RNN-model til EEG-baseret bevægelsesdetektion.
    
    Parameters:
    -----------
    input_shape : Tuple med formatet (window_size, num_channels)
    num_classes : Antal klasser (default: 2 - bevægelse/ingen bevægelse)
    
    Returns:
    --------
    model : Kompileret Keras-model
    """
    # Input lag, som tager EEG-data (vinduesstørrelser, kanaler)
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Reshape til RNN hvis nødvendigt - sikrer at dimensioner er klare
    # RNN forventer (batch_size, timesteps, features)
    x = tf.keras.layers.Reshape(input_shape)(inputs)
    
    # RNN del med flere LSTM-lag for at fange tidsmæssige sammenhænge
    # Første lag returnerer sekvenser for at kunne stables
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Andet LSTM-lag med return_sequences=True for at kunne stables
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Tredje LSTM-lag, som ikke returnerer sekvenser
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Dense layers til klassifikation
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Softmax-output: Klassificerer mellem bevægelse (1) og ingen bevægelse (0)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Returnerer den færdige LSTM-RNN-model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Print model summary for at verificere modellens struktur
    model.summary()
    
    return model

def train_and_evaluate_kfold(eeg_list, emg_list, subject_ids, window_size=500, epochs=100, batch_size=32, n_splits=5):
    """
    Træner og evaluerer modellen ved at bruge K-Fold krydsvalidering
    og registrerer træningstiden for hver fold og samlet.
    """
    print(f"Påbegynder træningspipeline med {n_splits}-Fold Cross-Validation (LSTM-RNN Model)")

    # Saml alt data
    all_X = []
    all_y = []
    
    total_start_time = time.time()

    # Forbereder data for hver person
    for i, (eeg_raw, emg_path, subject_id) in enumerate(zip(eeg_list, emg_list, subject_ids)):
        print(f"Forbereder data for {subject_id}...")
        try:
            emg_data = np.loadtxt(emg_path)
            
            # For hver forsøgsperson forbereder vi data
            X, y, n_channels, _, _ = prepare_data(eeg_raw, emg_data, window_size=window_size)
            
            # Ensure we have data
            if X.shape[0] == 0:
                print(f"Advarsel: Ingen data for forsøgsperson {subject_id}, springer over.")
                continue
                
            # Ensure X is in float32 format to avoid errors with TensorFlow
            X = X.astype(np.float32)
            
            # Convert y to int32
            y = y.astype(np.int32)
            
            # Add a check to ensure data has expected dimensions
            print(f"Data shape for {subject_id}: {X.shape}, labels shape: {y.shape}")
            
            all_X.append(X)
            all_y.append(y)
            
        except Exception as e:
            print(f"Fejl ved forberedelse af data for {subject_id}: {str(e)}")
            continue
    
    if not all_X:
        raise ValueError("Ingen gyldig data fundet for nogen forsøgspersoner.")
    
    # Kombiner alle data til ét stort dataset
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"Samlet data shape: {X.shape}, labels shape: {y.shape}")
    
    # Get the correct number of channels
    num_channels = X.shape[2]
    print(f"Number of channels detected: {num_channels}")
    
    # Initialiser K-Fold cross-validator
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Samler resultater fra hver fold
    fold_results = []
    fold_times = []
    all_confusion_matrices = []
    all_auc_scores = []
    all_fpr = []
    all_tpr = []

    plt.figure(figsize=(15, 10))

    # K-Fold Cross-Validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        fold_start_time = time.time()
        print(f"\nTræner fold {fold+1}/{n_splits}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Print shapes for debugging
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Laver og kompilerer LSTM-RNN modellen
        print("Creating LSTM-RNN model...")
        model = create_lstm_rnn_model(input_shape=(window_size, num_channels))
        
        # Using a different learning rate for RNN
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(f'rnn_model_fold{fold+1}_{timestamp}.keras',
                                               save_best_only=True,
                                               monitor='val_loss'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=15,  # Increased patience for RNN
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                factor=0.5,
                                                patience=5,
                                                min_lr=0.00001)
        ]

        # Træner modellen
        print("Starting training...")
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
            
        # Måler træningstid for denne fold
        fold_end_time = time.time()
        fold_training_time = fold_end_time - fold_start_time
        fold_times.append(fold_training_time)
        print(f"Fold {fold+1} træningstid: {fold_training_time:.2f} sekunder ({fold_training_time/60:.2f} minutter)")

        # Beregning af performance-metrikker
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        all_confusion_matrices.append(cm)

        # ROC Curve and AUC Score
        # Ensure we have predictions for class 1 (may need to adjust if your model is multi-class)
        if y_pred.shape[1] > 1:
            class1_preds = y_pred[:, 1]
        else:
            class1_preds = y_pred.flatten()
            
        fpr, tpr, _ = roc_curve(y_test, class1_preds)
        auc_score = auc(fpr, tpr)
        all_auc_scores.append(auc_score)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

        # Gemmer den højeste validation accuracy for denne fold
        val_accuracy = max(history.history['val_accuracy'])
        fold_results.append(val_accuracy)

        # Plot ROC kurve for denne fold
        plt.subplot(2, 3, 1)
        plt.plot(fpr, tpr, label=f'Fold {fold+1} (AUC = {auc_score:.2f})')
        
        print(f"Fold {fold+1} test accuracy: {val_accuracy:.4f}, AUC: {auc_score:.4f}")
        print("Confusion matrix:")
        print(cm)
        
        # Plot learning curves
        plt.subplot(2, 3, 4)
        plt.plot(history.history['accuracy'], label=f'Fold {fold+1} training')
        plt.plot(history.history['val_accuracy'], label=f'Fold {fold+1} validation')
        plt.title('Accuracy under træning')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        if fold == n_splits-1:  # Only add legend on the last fold
            plt.legend()

    # Beregn samlet træningstid
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"\nSamlet træningstid: {total_training_time:.2f} sekunder ({total_training_time/60:.2f} minutter)")

    if not all_confusion_matrices:
        raise ValueError("Ingen folds blev fuldført. Kontrollér data og model.")
        
    # Beregn og plot gennemsnitlig confusion matrix
    avg_cm = np.mean(all_confusion_matrices, axis=0)
    plt.subplot(2, 3, 2)
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Gennemsnitlig Confusion Matrix (LSTM-RNN)')
    plt.xlabel('Forudsagt')
    plt.ylabel('Sand')

    # Færdiggør ROC kurve plot
    plt.subplot(2, 3, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Kurver - LSTM-RNN Model')
    plt.legend(loc="lower right")

    # Plot træningtider per fold
    plt.subplot(2, 3, 3)
    bars = plt.bar(range(1, n_splits+1), [t/60 for t in fold_times])
    plt.title('Træningstid per fold')
    plt.xlabel('Fold')
    plt.ylabel('Tid (minutter)')
    plt.xticks(range(1, n_splits+1))
    # Tilføj tid-labels
    for bar, time_val in zip(bars, fold_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f'{time_val/60:.1f}m', ha='center', va='bottom')

    # Plot sammenfatning af metrikker
    plt.subplot(2, 3, 5)
    plt.axis('off')
    summary_text = f'LSTM-RNN Performance Metrics:\n\n' \
                   f'Gennemsnitlig Accuracy: {np.mean(fold_results):.4f}\n' \
                   f'Gennemsnitlig AUC: {np.mean(all_auc_scores):.4f}\n' \
                   f'Std Dev Accuracy: {np.std(fold_results):.4f}\n' \
                   f'Std Dev AUC: {np.std(all_auc_scores):.4f}\n\n' \
                   f'Samlet træningstid: {total_training_time/60:.2f} minutter\n' \
                   f'Gns. træningstid per fold: {np.mean(fold_times)/60:.2f} minutter'
    plt.text(0.1, 0.5, summary_text, fontsize=12)
    
    # Plot fold accuracies
    plt.subplot(2, 3, 6)
    bars = plt.bar(range(1, n_splits+1), fold_results)
    plt.title('Validation Accuracy per fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(range(1, n_splits+1))
    # Tilføj accuracy-labels
    for bar, acc in zip(bars, fold_results):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'lstm_rnn_kfold_performance_metrics_{timestamp}.png')
    plt.show()

    avg_accuracy = np.mean(fold_results)
    avg_auc = np.mean(all_auc_scores)

    print(f"\nGennemsnitlig validation præcision over alle folds: {avg_accuracy:.4f}")
    print(f"Gennemsnitlig AUC score over alle folds: {avg_auc:.4f}")
    print(f"\nConfusion Matrix (gennemsnit over alle folds):")
    print(avg_cm)
    print(f"\nTraining times for each fold (in minutes):")
    for i, fold_time in enumerate(fold_times):
        print(f"Fold {i+1}: {fold_time/60:.2f} minutes")
    print(f"Total training time: {total_training_time/60:.2f} minutes")

    return avg_accuracy, avg_auc, avg_cm, fold_times, total_training_time

if __name__ == "__main__":
    try:
        print("Starting 5-Fold Cross-Validation training and evaluation...")
        avg_accuracy, avg_auc, avg_cm, fold_times, total_time = train_and_evaluate_kfold(
            eeg_list=EEG_DATA, 
            emg_list=EMG_DATA,
            subject_ids=SUBJECT_IDS,
            window_size=500,
            epochs=50,        
            batch_size=32,
            n_splits=5       # K=5 for 5-fold cross-validation
        )
        print("Training and evaluation completed successfully.")
        print(f"Average training time per fold: {np.mean(fold_times)/60:.2f} minutes")
        print(f"Total training time: {total_time/60:.2f} minutes")
    except Exception as e:
        print(f"Error during training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()