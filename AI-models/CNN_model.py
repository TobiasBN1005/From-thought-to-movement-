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

def create_cnn_model(input_shape, num_classes=2):
    """
    Opretter en CNN-model til EEG-baseret bevægelsesdetektion uden LSTM komponenter.
    
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
    
    # Ekstra konvolutionelle lag for at erstatte LSTM's sekventielle mønstergenkendelse
    conv4 = tf.keras.layers.Conv1D(filters=256, kernel_size=9, padding='same', activation='relu')(conv3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv4)
    
    # Global pooling i stedet for LSTM
    x = tf.keras.layers.GlobalAveragePooling1D()(conv4)
    
    # Dropout for at reducere overfitting
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Dense layers til klassifikation
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Softmax-output: Klassificerer mellem bevægelse (1) og ingen bevægelse (0)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Returnerer den færdige CNN-model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Print model summary to verify shapes
    model.summary()
    
    return model


def train_and_evaluate_kfold_with_inference(eeg_list, emg_list, subject_ids, n_splits=5, window_size=500, epochs=50, batch_size=32):
    """
    Trains and evaluates the model using K-fold cross-validation.
    Tracks and reports both training and inference time.

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
    avg_inference_time : Average inference time per sample
    """
    
    print(f"Starting {n_splits}-fold cross-validation training and evaluation with inference timing...")

    all_data = []
    
    # Forbedrer al data
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
    print(f"Total dataset shape - X: {X_all.shape}, y: {y_all.shape}")

    fold_results = []
    all_auc_scores = []
    fold_training_times = []
    fold_inference_times = []

    # Laver K-fold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_all)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")

        # Start træningstid
        fold_start_time = time.time()

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        # Laver og kompilerer model
        model = create_cnn_model(input_shape=(window_size, num_channels))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Trlner modellen
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

        # Stop træningstid
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_training_times.append(fold_duration)

        # Begynder inference tid 
        inference_start_time = time.time()
        y_pred = model.predict(X_test)  # Run inference
        inference_end_time = time.time()

        # Compute inference time per sample
        total_inference_time = inference_end_time - inference_start_time
        avg_inference_time_per_sample = total_inference_time / len(X_test)
        fold_inference_times.append(avg_inference_time_per_sample)

        # Evaluate performance
        y_pred_classes = np.argmax(y_pred, axis=1)
        auc_score = auc(*roc_curve(y_test, y_pred[:, 1])[:2])
        all_auc_scores.append(auc_score)
        accuracy = np.mean(y_pred_classes == y_test)
        fold_results.append(accuracy)

        print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Avg Inference Time per Sample: {avg_inference_time_per_sample:.6f} sec")

    avg_accuracy = np.mean(fold_results)
    avg_auc = np.mean(all_auc_scores)
    avg_inference_time = np.mean(fold_inference_times)

    # Display results
    results_df = pd.DataFrame({
        "Fold": range(1, n_splits + 1),
        "Accuracy": fold_results,
        "AUC": all_auc_scores,
        "Training Time (s)": fold_training_times,
        "Avg Inference Time (s/sample)": fold_inference_times
    })

    tools.display_dataframe_to_user(name="AI Model Training & Inference Timing", dataframe=results_df)

    print("\nFinal Results:")
    print(f"Avg Accuracy: {avg_accuracy:.4f}")
    print(f"Avg AUC: {avg_auc:.4f}")
    print(f"Avg Inference Time per Sample: {avg_inference_time:.6f} sec")

    return avg_accuracy, avg_auc, avg_inference_time

# Execute the function with inference time measurement
train_and_evaluate_kfold_with_inference(EEG_DATA, EMG_DATA, SUBJECT_IDS, n_splits=5, epochs=50, batch_size=32)


# ---------------------- Run Training ----------------------#
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