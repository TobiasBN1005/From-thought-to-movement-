import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.svm import SVC  # Import SVM classifier
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

#Funktion to at tage feaatures fra EEG for SVM
def extract_features(X):
    """
    Extract features from windowed EEG data for SVM classification.
    
    Parameters:
    -----------
    X : Windowed EEG data with shape (n_windows, window_size, channels)
    
    Returns:
    --------
    features : Extracted features with shape (n_windows, n_features)
    """
    n_windows, window_size, n_channels = X.shape
    features = np.zeros((n_windows, n_channels * 6))  # 6 features per channel
    
    for i in range(n_windows):
        feature_idx = 0
        for c in range(n_channels):
            channel_data = X[i, :, c]
            
            # Bereggner statiske features
            features[i, feature_idx] = np.mean(channel_data)  # Middel
            feature_idx += 1
            features[i, feature_idx] = np.std(channel_data)  # Standard afvigelsen
            feature_idx += 1
            features[i, feature_idx] = np.max(channel_data) - np.min(channel_data)  # Rækkevidde
            feature_idx += 1
            features[i, feature_idx] = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)  # IQR
            feature_idx += 1
            
            # Frekvensdomænde ddomæne features 
            fft_vals = np.abs(np.fft.rfft(channel_data))
            features[i, feature_idx] = np.mean(fft_vals)  # Middel frekvens
            feature_idx += 1
            features[i, feature_idx] = np.sum(fft_vals**2)  # Energi
            feature_idx += 1
    
    return features

def train_and_evaluate_kfold(eeg_list, emg_list, subject_ids, n_splits=5, window_size=500, batch_size=32):
    """
    Trains and evaluates the SVM model using K-fold cross-validation.
    Tracks and reports execution time.
    
    Parameters:
    -----------
    eeg_list : List of EEG data for each subject
    emg_list : List of EMG file paths for each subject
    subject_ids : List of subject identifiers
    n_splits : Number of folds for K-fold cross validation
    window_size : Size of each window in samples
    batch_size : Batch size for training (not used for SVM, kept for compatibility)
    
    Returns:
    --------
    avg_accuracy : Average accuracy across all folds
    avg_auc : Average AUC score across all folds
    """
    
    #Begynder overordende tid
    overall_start_time = time.time()
    
    print(f"Starting {n_splits}-fold cross-validation training and evaluation with SVM...")

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
    print(f"Number of channels detected: {num_channels}")
    print(f"Total dataset shape - X: {X_all.shape}, y: {y_all.shape}")
    
    # Extract features for SVM
    print("Extracting features for SVM...")
    features = extract_features(X_all)
    print(f"Extracted features shape: {features.shape}")
    
    fold_results = []
    all_confusion_matrices = []
    all_auc_scores = []
    fold_training_times = []  # Store time taken per fold
    
    # Laver K-fold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(features)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        
        # begynder tiden for denne fold
        fold_start_time = time.time()

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Laver og træner SVM model
        # Using probability=True to get probability estimates for ROC curve
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        
        #Begynder træning
        try:
            svm_model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error during training: {str(e)}")
            continue

        #Stopper tiden for denne fold
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_training_times.append(fold_duration)
        
        # Print træningstiden for denne fold
        minutes, seconds = divmod(fold_duration, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"\nFold {fold + 1} training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

        #Forudsigelse og evaluér
        y_pred_proba = svm_model.predict_proba(X_test)
        y_pred_classes = svm_model.predict(X_test)
        
        # Beregner gennemsnit
        accuracy = svm_model.score(X_test, y_test)
        fold_results.append(accuracy)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        all_confusion_matrices.append(cm)

        # ROC Curve and AUC Score
        # Get the probability for class 1
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        auc_score = auc(fpr, tpr)
        all_auc_scores.append(auc_score)

        print(f"Fold {fold + 1} test accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        print("Confusion matrix:")
        print(cm)
        
        # Laver klassifikations report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))

    if not all_confusion_matrices:
        raise ValueError("No folds were completed. Check data and model.")

    avg_accuracy = np.mean(fold_results)
    avg_auc = np.mean(all_auc_scores)
    std_accuracy = np.std(fold_results)

    # Stoper overordnede tid
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    
    # Print total træningstid
    minutes, seconds = divmod(overall_duration, 60)
    hours, minutes = divmod(minutes, 60)
    
    print("\n" + "="*50)
    print(f"K-FOLD CROSS-VALIDATION RESULTS WITH SVM ({n_splits} FOLDS)")
    print("="*50)
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")
    print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Print tid pr. fold
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
        print("Starting K-fold cross-validation training with SVM...")
        avg_accuracy, avg_auc = train_and_evaluate_kfold(
            eeg_list=EEG_DATA, 
            emg_list=EMG_DATA,
            subject_ids=SUBJECT_IDS,
            n_splits=5,  # Use 5-fold cross-validation
            window_size=500,
            batch_size=32  # Not used for SVM but kept for compatibility
        )
        print("Training and evaluation completed successfully.")
        print(f"Final average accuracy: {avg_accuracy:.4f}")
        print(f"Final average AUC: {avg_auc:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")