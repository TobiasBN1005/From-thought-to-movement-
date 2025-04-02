#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:31:52 2025

@author: tobiasbendix
"""

# ---------------------- Importering af relevante biblioteker ----------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold  # Changed from GroupKFold to KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import datetime
import mne
from mne.io import read_raw_brainvision

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
SUBJECT_IDS = ['AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D']  # Still tracking subject IDs for reference

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


#Laver en LSTM-CNN model
def create_lstm_cnn_model(input_shape, num_classes=2):
    """
    Opretter en LSTM-CNN-model til EEG-baseret bevægelsesdetektion.
    
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
    
    # Reshape for Conv1D if needed - make sure ndim is clear
    # Conv1D expects (batch_size, timesteps, features)
    x = tf.keras.layers.Reshape(input_shape)(inputs)
    
    # CNN del for feature ekstraktion
    # 1D konvolutions-lag med forskellige filter-størrelser for at fange forskellige mønstre
    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)

    conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)

    conv3 = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same', activation='relu')(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv3)

    # Dropout for at reducere overfitting
    x = tf.keras.layers.Dropout(0.3)(conv3)
    
    # LSTM del for at fange tidsmæssige sammenhænge i de ekstraherede features
    # Ensure we use proper TensorFlow tensors for LSTM
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)

    # Dense layers til klassifikation
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Softmax-output: Klassificerer mellem bevægelse (1) og ingen bevægelse (0)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Returnerer den færdige LSTM-CNN-model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    #Printer model resumé, for at sikre shapes er rigtige
    model.summary()
    
    return model


def train_and_evaluate_kfold(eeg_list, emg_list, subject_ids, window_size=500, epochs=100, batch_size=32, n_splits=5):
    """
    Træner og evaluerer modellen ved at bruge K-fold krydsvalidering
    """
    import time  # Import time module for timing
    
    #Begynder overordnede tid
    overall_start_time = time.time()
    
    print(f"Påbegynder træningspipeline med {n_splits}-fold Cross-Validation")

    # Forbered alle data på tværs af forsøgspersoner
    all_X = []
    all_y = []
    subject_indices = []  #Tracker hvilket subjekt hver sample hører til

    # Forbereder data for hver person
    for i, (eeg_raw, emg_path, subject_id) in enumerate(zip(eeg_list, emg_list, subject_ids)):
        print(f"Forbereder data for {subject_id}...")
        try:
            emg_data = np.loadtxt(emg_path)
            
            # For hver forsøgsperson forbereder vi data
            X, y, n_channels, _, _ = prepare_data(eeg_raw, emg_data, window_size=window_size)
            
            # ´´Sujrer der er data
            if X.shape[0] == 0:
                print(f"Advarsel: Ingen data for forsøgsperson {subject_id}, springer over.")
                continue
                
            # Sikrer X er i  float32 forma
            X = X.astype(np.float32)
            
            # Ændrer y to int32
            y = y.astype(np.int32)
            
            #Printer data shape for subjektet
            print(f"Data shape for {subject_id}: {X.shape}, channels: {n_channels}")
            
            #Tilføjer data til samlingen
            all_X.append(X)
            all_y.append(y)
            
            #Tracker hvilket subjekt, hver sample hører til
            subject_indices.extend([i] * X.shape[0])
            
        except Exception as e:
            print(f"Fejl ved forberedelse af data for {subject_id}: {str(e)}")
            continue
    
    if not all_X:
        raise ValueError("Ingen gyldig data fundet for nogen forsøgspersoner.")
    
    # Kombinerer all data
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_indices = np.array(subject_indices)
    
    #Får det rigtige antal af kanaler
    num_channels = X.shape[2]
    print(f"Number of channels detected: {num_channels}")
    print(f"Total dataset shape: {X.shape}, labels shape: {y.shape}")
    
    # Initerer KFold Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Samler resultater fra hver fold
    fold_results = []
    all_confusion_matrices = []
    all_auc_scores = []
    all_fpr = []
    all_tpr = []
    fold_times = []  # Gemmer træningstiden for hver fold

    plt.figure(figsize=(15, 5))

    # Begynder K-fold Cross-Validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Begynder tiden for denne fold
        fold_start_time = time.time()
        
        print(f"\nTester Fold {fold + 1}/{n_splits}")
        
        # Split data for denne fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        
        test_subjects = subject_indices[test_idx]
        unique_subjects = np.unique(test_subjects)
        unique_subject_names = [subject_ids[i] for i in unique_subjects if i < len(subject_ids)]
        print(f"Test fold contains data from subjects: {unique_subject_names}")
        
     
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Laver og kompilerer modellen
        print("Creating model...")
        model = create_lstm_cnn_model(input_shape=(window_size, num_channels))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(f'model_fold{fold+1}_{timestamp}.keras',
                                              save_best_only=True,
                                              monitor='val_loss'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=10,
                                           restore_best_weights=True)
        ]

        # Træner modellen
        print("Starting training...")
        try:
            training_start_time = time.time()  # Start timing the training process
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            training_end_time = time.time()  # Stopper træningstiden
            training_duration = training_end_time - training_start_time
            print(f"Training time for fold {fold + 1}: {training_duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            continue

        # Beregning af performance-metrikker
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        all_confusion_matrices.append(cm)

        # ROC Curve og AUC Score
        #Sikrer jeg har forudsigelser for klasse 1
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
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, label=f'Fold {fold+1} (AUC = {auc_score:.2f})')
        
        # Calculate and store fold time
        fold_end_time = time.time()
        fold_time = fold_end_time - fold_start_time
        fold_times.append(fold_time)
        
        print(f"Fold {fold+1} test accuracy: {val_accuracy:.4f}, AUC: {auc_score:.4f}")
        print(f"Total processing time for fold: {fold_time:.2f} seconds")
        print("Confusion matrix:")
        print(cm)

    #Beregner overordnede træningstid
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time

    if not all_confusion_matrices:
        raise ValueError("Ingen folds blev fuldført. Kontrollér data og model.")
        
    # Beregn og plot gennemsnitlig confusion matrix
    avg_cm = np.mean(all_confusion_matrices, axis=0)
    plt.subplot(1, 3, 2)
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Gennemsnitlig Confusion Matrix')
    plt.xlabel('Forudsagt')
    plt.ylabel('Sand')

    # Færdiggør ROC kurve plot
    plt.subplot(1, 3, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Kurver ({n_splits}-fold Cross-Validation)')
    plt.legend(loc="lower right")

    # Plot sammenfatning af metrikker
    plt.subplot(1, 3, 3)
    plt.axis('off')
    summary_text = f'Performance Metrics:\n\n' \
                   f'Gennemsnitlig Accuracy: {np.mean(fold_results):.4f}\n' \
                   f'Gennemsnitlig AUC: {np.mean(all_auc_scores):.4f}\n' \
                   f'Std Dev AUC: {np.std(all_auc_scores):.4f}'
    plt.text(0.1, 0.5, summary_text, fontsize=12)

    plt.tight_layout()
    plt.savefig(f'kfold{n_splits}_performance_metrics_{timestamp}.png')
    plt.show()

    avg_accuracy = np.mean(fold_results)
    avg_auc = np.mean(all_auc_scores)

    print(f"\nGennemsnitlig validation præcision over alle folds: {avg_accuracy:.4f}")
    print(f"Gennemsnitlig AUC score over alle folds: {avg_auc:.4f}")
    print(f"\nConfusion Matrix (gennemsnit over alle folds):")
    print(avg_cm)
    
    # Print timing information
    print("\n----- Training Time Summary -----")
    print(f"Total training process time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"Average time per fold: {np.mean(fold_times):.2f} seconds ({np.mean(fold_times)/60:.2f} minutes)")
    for i, fold_time in enumerate(fold_times):
        print(f"Fold {i+1}: {fold_time:.2f} seconds ({fold_time/60:.2f} minutes)")

    return avg_accuracy, avg_auc, avg_cm, fold_times, overall_time


# ---------------------- Påbegyndelse af træning ----------------------
avg_accuracy, avg_auc, avg_cm, fold_times, overall_time = train_and_evaluate_kfold(
    eeg_list=EEG_DATA, 
    emg_list=EMG_DATA, 
    subject_ids=SUBJECT_IDS, 
    window_size=500, 
    epochs=50, 
    batch_size=32,
    n_splits=5  #Antallet af folds for k-krydsvalidering
)