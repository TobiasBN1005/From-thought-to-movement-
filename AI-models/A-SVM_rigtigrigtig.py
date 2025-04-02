#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 19:37:14 2025

@author: tobiasbendix
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
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin

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
    Forbereder EEG og EMG data til træning af A-SVM-modellen.
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


def extract_features(X, feature_type='combined'):
    """
    Udtrækker features fra EEG-data til brug i SVM.
    
    Parameters:
    -----------
    X : Numpy array med EEG-vinduer (n_samples, window_size, n_channels)
    feature_type : Type af features der skal udtrækkes ('temporal', 'spectral', eller 'combined')
    
    Returns:
    --------
    features : Numpy array med udtrukne features (n_samples, n_features)
    """
    n_samples, window_size, n_channels = X.shape
    features_list = []
    
    # Beregn temporale features (tidsdomain)
    if feature_type in ['temporal', 'combined']:
        # Middelvælder for hvert vindue og kanal
        means = np.mean(X, axis=1)
        
        # Standardafvigelse for hvert vindue og kanal
        stds = np.std(X, axis=1)
        
        # Skævhed (skewness) for hvert vindue og kanal
        skewness = np.array([[np.mean(((x - np.mean(x)) / np.std(x))**3) if np.std(x) > 0 else 0 
                            for x in sample] for sample in X])
        
        # Kurtosis for hvert vindue og kanal
        kurtosis = np.array([[np.mean(((x - np.mean(x)) / np.std(x))**4) if np.std(x) > 0 else 0 
                            for x in sample] for sample in X])
        
        # Zero crossing rate for hvert vindue og kanal
        zcr = np.array([[np.sum(np.diff(np.signbit(x).astype(int)) != 0) / (len(x) - 1) 
                        for x in sample] for sample in X])
        
        # Føj temporale features til listen
        features_list.extend([means, stds, skewness, kurtosis, zcr])
    
    # Beregn spektrale features (frekvensdomæne)
    if feature_type in ['spectral', 'combined']:
        # Beregn power spectral density (PSD)
        from scipy import signal
        
        # Tomme arrays til at gemme spektrale features
        psd_delta = np.zeros((n_samples, n_channels))
        psd_theta = np.zeros((n_samples, n_channels))
        psd_alpha = np.zeros((n_samples, n_channels))
        psd_beta = np.zeros((n_samples, n_channels))
        psd_gamma = np.zeros((n_samples, n_channels))
        
        # Beregn PSD for hver sample og kanal
        for i in range(n_samples):
            for j in range(n_channels):
                freqs, psd = signal.welch(X[i, :, j], fs=250, nperseg=min(256, window_size))
                
                # Beregn gennemsnitlig power i hver frekvensgruppe
                delta_idx = np.where((freqs >= 0.5) & (freqs <= 4))
                theta_idx = np.where((freqs > 4) & (freqs <= 8))
                alpha_idx = np.where((freqs > 8) & (freqs <= 13))
                beta_idx = np.where((freqs > 13) & (freqs <= 30))
                gamma_idx = np.where((freqs > 30) & (freqs <= 45))
                
                psd_delta[i, j] = np.mean(psd[delta_idx]) if len(delta_idx[0]) > 0 else 0
                psd_theta[i, j] = np.mean(psd[theta_idx]) if len(theta_idx[0]) > 0 else 0
                psd_alpha[i, j] = np.mean(psd[alpha_idx]) if len(alpha_idx[0]) > 0 else 0
                psd_beta[i, j] = np.mean(psd[beta_idx]) if len(beta_idx[0]) > 0 else 0
                psd_gamma[i, j] = np.mean(psd[gamma_idx]) if len(gamma_idx[0]) > 0 else 0
        
        # Føj spektrale features til listen
        features_list.extend([psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma])
    
    # Saml og reshape alle features
    all_features = np.concatenate([f.reshape(n_samples, -1) for f in features_list], axis=1)
    
    return all_features


class AdaptiveSVM(BaseEstimator, ClassifierMixin):
    """
    Adaptiv SVM til EEG-baseret bevægelsesdetektion.
    Tilpasser sig skiftende fordelinger i EEG-data ved at kombinere en base SVM
    med en adaptiv komponent.
    """
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', adaptation_rate=0.2):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.adaptation_rate = adaptation_rate
        self.base_svm = None
        self.adaptive_svm = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Træner base SVM på træningsdata.
        
        Parameters:
        -----------
        X : Features udtrukket fra EEG-data
        y : Klasse-labels (0: ingen bevægelse, 1: bevægelse)
        
        Returns:
        --------
        self : Trænet model
        """
        # Træn base SVM
        self.base_svm = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True)
        self.base_svm.fit(X, y)
        
        # Initialisér adaptiv SVM (den adaptive komponent)
        self.adaptive_svm = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True)
        
        # Marker at modellen er trænet
        self.is_fitted = True
        return self
    
    def partial_fit(self, X, y):
        """
        Opdaterer den adaptive komponent med nye data.
        
        Parameters:
        -----------
        X : Nye features udtrukket fra EEG-data
        y : Nye klasse-labels
        
        Returns:
        --------
        self : Opdateret model
        """
        if not self.is_fitted:
            return self.fit(X, y)
        
        # Træn adaptiv SVM på nye data
        self.adaptive_svm.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Forudsiger klasse-labels ved at kombinere base og adaptiv SVM.
        
        Parameters:
        -----------
        X : Features udtrukket fra EEG-data
        
        Returns:
        --------
        y_pred : Forudsagte klasse-labels
        """
        if not self.is_fitted:
            raise ValueError("Model skal trænes med fit() før predict() kan bruges")
        
        # Basisforudsigelser fra base SVM
        base_pred = self.base_svm.predict_proba(X)
        
        # Hvis adaptiv SVM er trænet, kombiner forudsigelser
        if hasattr(self.adaptive_svm, 'classes_'):
            adapt_pred = self.adaptive_svm.predict_proba(X)
            
            # Vægtede kombinationer af forudsigelser
            combined_pred = (1 - self.adaptation_rate) * base_pred + self.adaptation_rate * adapt_pred
            return np.argmax(combined_pred, axis=1)
        else:
            # Brug kun base SVM, hvis adaptiv SVM ikke er trænet
            return self.base_svm.predict(X)
    
    def predict_proba(self, X):
        """
        Forudsiger klasseandsynligheder ved at kombinere base og adaptiv SVM.
        
        Parameters:
        -----------
        X : Features udtrukket fra EEG-data
        
        Returns:
        --------
        y_proba : Forudsagte klasseandsynligheder
        """
        if not self.is_fitted:
            raise ValueError("Model skal trænes med fit() før predict_proba() kan bruges")
        
        # Basisklasse-sandsynligheder fra base SVM
        base_proba = self.base_svm.predict_proba(X)
        
        # Hvis adaptiv SVM er trænet, kombiner sandsynligheder
        if hasattr(self.adaptive_svm, 'classes_'):
            adapt_proba = self.adaptive_svm.predict_proba(X)
            
            # Vægtede kombinationer af sandsynligheder
            combined_proba = (1 - self.adaptation_rate) * base_proba + self.adaptation_rate * adapt_proba
            return combined_proba
        else:
            # Brug kun base SVM, hvis adaptiv SVM ikke er trænet
            return base_proba


def train_and_evaluate_kfold(eeg_list, emg_list, subject_ids, n_splits=5, window_size=500):
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
    
    Returns:
    --------
    avg_accuracy : Average accuracy across all folds
    avg_auc : Average AUC score across all folds
    """
    
    # Starter overordnede tid
    overall_start_time = time.time()
    
    print(f"Starting {n_splits}-fold cross-validation training and evaluation with A-SVM...")

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
            
            # Udtræk features fra EEG-vinduerne
            print(f"Extracting features for {subject_id}...")
            X_features = extract_features(X, feature_type='combined')
            
            all_data.append((X_features.astype(np.float32), y.astype(np.int32), subject_id))
            
        except Exception as e:
            print(f"Error processing {subject_id}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No valid data found for any subject.")

    # Concatenate al data
    X_all = np.concatenate([data[0] for data in all_data], axis=0)
    y_all = np.concatenate([data[1] for data in all_data], axis=0)
    
    print(f"Total feature dataset shape - X: {X_all.shape}, y: {y_all.shape}")
    
    fold_results = []
    all_confusion_matrices = []
    all_auc_scores = []
    fold_training_times = []  # Gemmer tid taget pr. fold
    
    # Laver K-fold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_all)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        
        # Start timing this fold
        fold_start_time = time.time()

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        print(f"Training set: {X_train.shape} samples, Test set: {X_test.shape} samples")
        
        # Standardisering af feature-data (vigtig for SVM)
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)

        # Oprettelse og træning af Adaptiv SVM
        print("Training Adaptive SVM...")
        a_svm = AdaptiveSVM(C=1.0, kernel='rbf', gamma='scale', adaptation_rate=0.2)
        a_svm.fit(X_train_scaled, y_train)
        
        # Simulér online adaptation (for at teste adaptiv komponent)
        print("Simulating adaptation with a portion of test data...")
        # Brug 20% af testdata til adaptation
        adapt_size = int(0.2 * len(X_test_scaled))
        if adapt_size > 0:
            a_svm.partial_fit(X_test_scaled[:adapt_size], y_test[:adapt_size])
            # Evaluer på de resterende 80%
            X_test_final = X_test_scaled[adapt_size:]
            y_test_final = y_test[adapt_size:]
        else:
            X_test_final = X_test_scaled
            y_test_final = y_test

        # Stop tiden for denne fold
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_training_times.append(fold_duration)
        
        # Print træningstid for denne fold
        minutes, seconds = divmod(fold_duration, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"\nFold {fold + 1} training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

        # Forudsigelser og evaluering
        print("Evaluating model performance...")
        y_pred_proba = a_svm.predict_proba(X_test_final)
        y_pred_classes = a_svm.predict(X_test_final)

        # Confusion Matrix
        cm = confusion_matrix(y_test_final, y_pred_classes)
        all_confusion_matrices.append(cm)

        # Beregn accuracy
        accuracy = np.mean(y_pred_classes == y_test_final)
        fold_results.append(accuracy)

        # ROC Curve and AUC Score
        if y_pred_proba.shape[1] > 1:
            class1_preds = y_pred_proba[:, 1]
        else:
            class1_preds = y_pred_proba.flatten()
            
        fpr, tpr, _ = roc_curve(y_test_final, class1_preds)
        auc_score = auc(fpr, tpr)
        all_auc_scores.append(auc_score)

        print(f"Fold {fold + 1} test accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        print("Confusion matrix:")
        print(cm)
        
        # Laver klassifikations report
        print("\nClassification Report:")
        print(classification_report(y_test_final, y_pred_classes))

    if not all_confusion_matrices:
        raise ValueError("No folds were completed. Check data and model.")

    avg_accuracy = np.mean(fold_results)
    avg_auc = np.mean(all_auc_scores)
    std_accuracy = np.std(fold_results)

    # Stop overordnede tid
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    
    # Print total træningstid
    minutes, seconds = divmod(overall_duration, 60)
    hours, minutes = divmod(minutes, 60)
    
    print("\n" + "="*50)
    print(f"K-FOLD CROSS-VALIDATION RESULTS WITH A-SVM ({n_splits} FOLDS)")
    print("="*50)
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")
    print("\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
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
        print("Starting K-fold cross-validation training with Adaptive SVM...")
        avg_accuracy, avg_auc = train_and_evaluate_kfold(
            eeg_list=EEG_DATA, 
            emg_list=EMG_DATA,
            subject_ids=SUBJECT_IDS,
            n_splits=5,  # Use 5-fold cross-validation
            window_size=500
        )
        print("Training and evaluation completed successfully.")
        print(f"Final average accuracy: {avg_accuracy:.4f}")
        print(f"Final average AUC: {avg_auc:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")