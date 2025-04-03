#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:06:14 2025

@author: tobiasbendix
"""
"""
Første forsøg på en model
"""

#Fil: LeModel

"""
Bevægelses-intention-detektor til projektet "Fra tanke til bevægelse"
Modellen implementerer en dyb lærings pipeline ved at bruge en multimodal transformer,
til at forudsige ønskede/intentionelle bevægelser, baseret på EEG, EMG og IMU data. I koden er:
    
1. Import: Importering af biblioteker
2. TimingMetrics: En "hjælpeklasse" som måler trænings- og inferenstiden.
3. PositionalEncoding: Et modul som tilføjer positional embeddings.
4. MultimodalTransformer: En transformer-model med 3 modalitetsgrene: EEG, EMG og IMU.
5. MovementIntentionDetector: En wrapper til at forbehandle signaler, træne modellen og køre inferens.
6. Evalueringsfunktioner: Funktioner der evaluerer modellens præstation.
7. Main function: Her køres hele funktion. Simulerede data skal udskiftes med mit eget. 


Modellens output er binære signaler til forskellige bevægelsetyper, som skal sendes til et Arudino board,
der tolker signalerne som bevægelser på en robotarm. På den måde tolkes ens tanke som en bevægelse. 
"""

# ============================================================
# 1. Import af biblioteker
# ============================================================
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# ============================================================
# 2. Tidsmetrikker
# ============================================================
class TimingMetrics: #Måler trænings-inferenstider
    def __init__(self):
        self.training_times = []
        self.inference_times = []
   
    def add_training_time(self, duration):
        self.training_times.append(duration)
    
    def add_inference_time(self, duration):
        self.inference_times.append(duration)
    
    def get_training_stats(self):
        return {
            'middel_træningstid': np.mean(self.training_times),
            'min_træningstid': np.min(self.training_times),
            'max_træningstid': np.max(self.training_times),
            'std_træningstid': np.std(self.training_times)
        }
        
    def get_inference_stats(self):
        return {
            'mean_inferenstid': np.mean(self.inference_times),
            'min_inferenstid': np.min(self.inference_times),
            'max_inferenstid': np.max(self.inference_times),
            'std_inferenstid': np.std(self.inference_times)
        }
    
    def print_summary(self):
        print("\n--- Tidsmetrik resumé ---")
        print("Træningstider:")
        training_stats = self.get_training_stats()
        for key, value in training_stats.items():
            print(f"{key}: {value:.4f} sekunder")
        print("\nInferenstider:")
        inference_stats = self.get_inference_stats()
        for key, value in inference_stats.items():
            print(f"{key}: {value:.4f} sekunder")
# ============================================================
# 3. Positiontal Encoding
# ============================================================
class PositionalEncoding(nn.Module): #Tilføjer tidspositions-information, noget der mangler i en transformer
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
    # x shape: (batch, time_steps, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x    
# ============================================================
# 4. Multimodal Transformer-model
# ============================================================
class MultimodalTransformer(nn.Module):
    def __init__(self, 
        eeg_channels, emg_channels, imu_channels,
        time_steps,
        d_model=64,
        num_heads=4,
        num_transformer_layers=3,
        num_movements=5,
        dropout=0.3):
        super(MultimodalTransformer, self).__init__()
        
        # ----- EEG-gren -----
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(in_channels=eeg_channels, out_channels=64, kernel_size=3, padding=1), #Konvulationslag -> reducerer dimensionalitet
            nn.BatchNorm1d(64),         #normaliserer signaler
            nn.ReLU(),                  #introducerer ikke-linearitet
            nn.MaxPool1d(kernel_size=2) #educerer tidsopløsning
        )
        self.eeg_proj = nn.Linear(64, d_model)
    
        # ----- EMG-gren -----
        self.emg_conv = nn.Sequential(
            nn.Conv1d(in_channels=emg_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.emg_proj = nn.Linear(32, d_model)
    
        # ----- IMU-gren -----
        self.imu_conv = nn.Sequential(
            nn.Conv1d(in_channels=imu_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.imu_proj = nn.Linear(32, d_model)
        
        # Beregner ny tidsdimension efter pooling
        self.pool_factor = 2  
        new_time_steps = time_steps // self.pool_factor
    
        # ---- Positional Encoding ----
        # Jeg ganger max_len med, fordi jeg vil concatenate 3 modaliteter over tid
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=new_time_steps * 3)
        
        # ---- Transformer Encoder ---- #behandler de komplekse signaler, kan fange afhængigheder mellem signalerne
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True  # Add this parameter
        )
        
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        
        # ---- Klassifikations-hoved ----
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_movements)
        )
    
    def forward(self, eeg, emg, imu):
        # Input-shapes: (batch, channels, time)
        
        #Behandler hver modalitet. FørstEEG-gren
        eeg_feat = self.eeg_conv(eeg)  # shape: (batch, 64, new_time)
        eeg_feat = eeg_feat.permute(0, 2, 1)  # (batch, new_time, 64)
        eeg_feat = self.eeg_proj(eeg_feat)    # (batch, new_time, d_model)
        
        #EMG-gren
        emg_feat = self.emg_conv(emg)
        emg_feat = emg_feat.permute(0, 2, 1)
        emg_feat = self.emg_proj(emg_feat)
        
        #IMU-gren
        imu_feat = self.imu_conv(imu)
        imu_feat = imu_feat.permute(0, 2, 1)
        imu_feat = self.imu_proj(imu_feat)
        
        #Kombinerer signalerne langs tidsdimensionen
        combined = torch.cat([eeg_feat, emg_feat, imu_feat], dim=1)  # shape: (batch, total_time, d_model)
        combined = self.pos_encoding(combined)
       
        # No need to permute with batch_first=True
        transformer_out = self.transformer_encoder(combined)
       
        #Gennemsnitspool og klassificering
        pooled = transformer_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
# ============================================================
# 5. Bevægelse-intention-detektor
# ============================================================
class MovementIntentionDetector: #Wrapper til træning og inferens
    def __init__(self, sampling_rate=500, time_steps=1000,
        eeg_channels=64, emg_channels=1, imu_channels=6):
        
        """
        Initerer detektionsparametre.
        """
        
        self.sampling_rate = sampling_rate
        self.time_steps = time_steps
        self.eeg_channels = eeg_channels
        self.emg_channels = emg_channels
        self.imu_channels = imu_channels
        self.model = None
        self.timing_metrics = TimingMetrics()
            
    def bandpass_filter(self, signal_data, lowcut=1, highcut=40):
        """
        Anvender båndpass fiter, for at filtre inputsignalet
        
        Parametre:
        - signal_data: Input signal som skal filtrees
        - lowcut: Lower cutoff frequency (Hz)
        - highcut: Higher cutoff frequency (Hz)
        
        Retunerer:
        - Filteret signal
        """
        #Sikerer signalet er numpy array
        signal_data = np.asarray(signal_data)
        
        #Nyquist-frekvens er den halve samplingrate
        nyquist = 0.5 * self.sampling_rate
        
        #Normaliserer frekvenser
        low = lowcut / nyquist
        high = highcut / nyquist
        
        #Sikrer frekvenser er indefor den valide rækkevidde
        if low <= 0:
            low = 0.001
        if high >= 1:
            high = 0.999
        
        #Laver et Butterworth båndpas filter
        b, a = signal.butter(N=6, Wn=[low, high], btype='band')
        
        #Anvender filter
        if signal_data.ndim == 2:
            filtered = np.array([signal.filtfilt(b, a, sig) for sig in signal_data])
        else:
            filtered = signal.filtfilt(b, a, signal_data)
    
        return filtered
        
    def preprocess_signals(self, eeg_data, emg_data, imu_data):
        
        """
        Anvender båndpasfiltrering og ændrer signaler til torch tensorer.
        Hvis input har shape (channels, time), tilføjes en en batch dimension
        """
        
        eeg_filtered = self.bandpass_filter(eeg_data)
        emg_filtered = self.bandpass_filter(emg_data, lowcut=10, highcut=250)
        imu_filtered = imu_data  #Antaget at IMU er forberedet på forhånden
        
        #Chekker dimensioner: Hvis 2D, så unsqueeze; hvis 3D, antag givet batch 
        if eeg_filtered.ndim == 2:
            eeg_tensor = torch.FloatTensor(eeg_filtered).unsqueeze(0)
        else:
            eeg_tensor = torch.FloatTensor(eeg_filtered)
        if emg_filtered.ndim == 2:
            emg_tensor = torch.FloatTensor(emg_filtered).unsqueeze(0)
        else:
            emg_tensor = torch.FloatTensor(emg_filtered)
        if imu_filtered.ndim == 2:
            imu_tensor = torch.FloatTensor(imu_filtered).unsqueeze(0)
        else:
            imu_tensor = torch.FloatTensor(imu_filtered)
            
        return eeg_tensor, emg_tensor, imu_tensor
    
    def train(self, eeg_data, emg_data, imu_data, movement_labels, num_epochs=50, class_weights=None):
        """
        Træner modellen ved at bruge givne signaler og labels.
        Understøtter valgfri klassevægte for håndtering af ubalancerede datasæt.
        
        Args:
        - eeg_data: EEG signaler
        - emg_data: EMG signaler
        - imu_data: IMU signaler
        - movement_labels: Bevægelseslabels
        - num_epochs: Antal træningsepoch
        - class_weights: Valgfrie klassevægte for tab-funktion
        """
        
        # Start tidsmåling
        start_time = time.time()
        
        # Forbehandl signaler
        eeg_tensor, emg_tensor, imu_tensor = self.preprocess_signals(eeg_data, emg_data, imu_data)
        
        # Sikrer movement_labels er 2D: (batch, num_movements)
        if movement_labels.ndim == 1:
            movement_labels = movement_labels[np.newaxis, :]
        labels_tensor = torch.FloatTensor(movement_labels)
        
        # Initerer model
        self.model = MultimodalTransformer(
            eeg_channels=self.eeg_channels,
            emg_channels=self.emg_channels,
            imu_channels=self.imu_channels,
            time_steps=self.time_steps,
            d_model=64,
            num_heads=4,
            num_transformer_layers=3,
            num_movements=labels_tensor.size(1),
            dropout=0.3
        )
        
        # Vælg tab-funktion
        if class_weights is not None:
            # Konverter klassevægte til tensor
            class_weights = torch.FloatTensor(class_weights)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            # Standard BCEWithLogitsLoss uden vægte
            criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer med adaptive learning rate
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-3,  # Basis learning rate
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Forbered til træning
        epoch_times = []
        best_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 10
        
        # Træningsloop med forbedret overvågning
        self.model.train()
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Nulstil gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(eeg_tensor, emg_tensor, imu_tensor)
            
            # Beregn tab
            loss = criterion(outputs, labels_tensor)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for at undgå eksploderende gradienter
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Opdater vægte
            optimizer.step()
            
            # Opdater learning rate baseret på tab
            scheduler.step(loss)
            
            # Beregn epoch tid
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Udskriv træningsdetaljer
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Epoch Tid: {epoch_time:.4f} s")
            
            # Tidlig stop
            if loss.item() < best_loss:
                best_loss = loss.item()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Stop træning hvis ingen forbedring
            if early_stopping_counter >= early_stopping_patience:
                print(f"Tidlig stop efter {epoch+1} epochs")
                break
        
        # Beregn total træningstid
        total_training_time = time.time() - start_time
        self.timing_metrics.add_training_time(total_training_time)
        
        # Returner træningsstatistikker
        return {
            'total_træningstid': total_training_time,
            'middel_epoch_tid': np.mean(epoch_times),
            'std_epoch_tid': np.std(epoch_times),
            'bedste_tab': best_loss
    }
    
    def predict_movement_intention(self, eeg_data, emg_data, imu_data):
        
        """
        Kører inference på en ny sample, og retunerer binære forudsigelser. 
        """
        
        if self.model is None:
            raise ValueError("Model ikke trænet Kald train() først.")
        start_time = time.time()
        self.model.eval()
        eeg_tensor, emg_tensor, imu_tensor = self.preprocess_signals(eeg_data, emg_data, imu_data)
        with torch.no_grad():
            logits = self.model(eeg_tensor, emg_tensor, imu_tensor)
        inference_time = time.time() - start_time
        self.timing_metrics.add_inference_time(inference_time)
        probs = torch.sigmoid(logits)
        binary_signals = (probs > 0.5).int()
        return binary_signals, inference_time
# ============================================================
# 6. Evalueringsfunktioner
# ============================================================
def compute_confusion_matrix(y_true, y_pred):
    """
    Laver en confusion matrix, med rigtige og forudsagte binære labels.
    For multi-label klassifikation, lav en pr. label.
    """
    
    num_labels = y_true.shape[1]
    cm_total = {}
    for i in range(num_labels):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        cm_total[f'Label_{i}'] = cm
    return cm_total
        
def plot_roc_curves(y_true, y_scores, num_labels):
        
        """
        Plotter ROC-kurver og beregner AUC-score for hver markering
        Plot ROC curves and compute AUC scores for each label.
        y_true: numpy array af shape (samples, num_labels)
        y_scores: Forudsagte scores (sandsynligheder) af shape (samples, num_labels)
        """
        
        plt.figure(figsize=(10, 8))
        auc_scores = {}
        for i in range(num_labels):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            auc_score = auc(fpr, tpr)
            auc_scores[f'Label_{i}'] = auc_score
            plt.plot(fpr, tpr, label=f'Label {i} (AUC = {auc_score:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curves')
        plt.xlabel('Falsk Positiv Ratio')
        plt.ylabel('Rigtig Positiv Ratio')
        plt.legend()
        plt.grid(True)
        plt.show()
        return auc_scores
    
def run_kfold_cv(X_eeg, X_emg, X_imu, y, num_folds=5, num_epochs=50):
    """
    Kør K-fold kryds-validering med forbedret databehandling og evaluering.
    
    Args:
    - X_eeg, X_emg, X_imu: Input signaler (samples, channels, time)
    - y: Binære labels (samples, num_movements)
    - num_folds: Antal fold i cross-validation
    - num_epochs: Antal træningsepoch per fold
    
    Returns:
    - Gennemsnitlig præcision
    - Gennemsnitlige AUC scores
    - Confusion matrices
    """
    
    # Tilføj funktioner til databehandling
    def normalize_signals(X):
        """
        Normalisér signaler ved z-score normalisering
        """
        X_normalized = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                channel = X[i, j, :]
                X_normalized[i, j, :] = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)
        return X_normalized
    
    def compute_class_weights(y):
        """
        Beregn klassevægte for at håndtere ubalancerede datasæt
        """
        class_counts = np.sum(y, axis=0)
        total_samples = y.shape[0]
        weights = total_samples / (len(np.unique(y)) * class_counts)
        return torch.FloatTensor(weights)
    
    # Normalisér input signaler
    X_eeg = normalize_signals(X_eeg)
    X_emg = normalize_signals(X_emg)
    X_imu = normalize_signals(X_imu)
    
    # Opsæt K-fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Resultat-beholdere
    all_accuracy = []
    all_auc = {f'Label_{i}': [] for i in range(y.shape[1])}
    all_cm = []
    
    # K-fold loop
    for fold, (train_index, test_index) in enumerate(kf.split(X_eeg), 1):
        print(f"\n--- Fold {fold} ---")
        
        # Opdel data
        X_eeg_train, X_eeg_test = X_eeg[train_index], X_eeg[test_index]
        X_emg_train, X_emg_test = X_emg[train_index], X_emg[test_index]
        X_imu_train, X_imu_test = X_imu[train_index], X_imu[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Beregn klassevægte
        class_weights = compute_class_weights(y_train)
        
        # Opret detektor
        detector = MovementIntentionDetector(
            sampling_rate=500,
            time_steps=X_eeg_train.shape[2],
            eeg_channels=X_eeg_train.shape[1],
            emg_channels=X_emg_train.shape[1],
            imu_channels=X_imu_train.shape[1]
        )
        
        # Træning med gennemsnit af træningsdata
        eeg_train_mean = np.mean(X_eeg_train, axis=0)
        emg_train_mean = np.mean(X_emg_train, axis=0)
        imu_train_mean = np.mean(X_imu_train, axis=0)
        y_train_mean = np.mean(y_train, axis=0)
        
        # Modificér træningsmetoden til at bruge klassevægte
        detector.train(
            eeg_train_mean, 
            emg_train_mean, 
            imu_train_mean, 
            y_train_mean, 
            num_epochs=num_epochs,
            class_weights=class_weights  # Tilføj denne parameter til train-metoden
        )
        
        # Forudsigelser for testdata
        y_preds = []
        y_scores = []
        
        for i in range(X_eeg_test.shape[0]):
            # Forudsig bevægelsesintention
            pred, _ = detector.predict_movement_intention(
                X_eeg_test[i], 
                X_emg_test[i], 
                X_imu_test[i]
            )
            
            # Få sandsynlighedsscorer
            detector.model.eval()
            with torch.no_grad():
                eeg_tensor, emg_tensor, imu_tensor = detector.preprocess_signals(
                    X_eeg_test[i], 
                    X_emg_test[i], 
                    X_imu_test[i]
                )
                logits = detector.model(eeg_tensor, emg_tensor, imu_tensor)
                probs = torch.sigmoid(logits).numpy()
            
            y_preds.append((probs > 0.5).astype(int))
            y_scores.append(probs)
        
        # Konverter forudsigelser
        y_preds = np.vstack(y_preds)
        y_scores = np.vstack(y_scores)
        
        # Beregn præcision
        acc = np.mean((y_preds == y_test).astype(int))
        all_accuracy.append(acc)
        
        # Beregn AUC for hver label
        aucs = {}
        for i in range(y.shape[1]):
            # Håndter tilfælde med ingen positive prøver
            if np.sum(y_test[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_test[:, i], y_scores[:, i])
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nan
            
            aucs[f'Label_{i}'] = auc_score
            all_auc[f'Label_{i}'].append(auc_score)
        
        # Beregn confusion matrix
        cms = compute_confusion_matrix(y_test, y_preds)
        all_cm.append(cms)
        
        # Udskriv resultater for dette fold
        print(f"Fold {fold} Præcision: {acc:.4f}")
        print(f"Fold {fold} AUC scores: {aucs}")
    
    # Aggreger resultater
    avg_acc = np.mean(all_accuracy)
    avg_auc = {label: np.nanmean(scores) for label, scores in all_auc.items()}
    
    # Udskriv resumé
    print("\n--- Cross-Validation Resumé ---")
    print(f"Gennemsnitspræcision: {avg_acc:.4f}")
    print(f"Gennemsnitlig AUC scores per label: {avg_auc}")
    
    return avg_acc, avg_auc, all_cm
# ============================================================
# 7. Mainfunktion
# ============================================================
def main():
    
    #Simulationsparametre for en single sample trænings/inferens demonstration SKAL ÆNDRES
    time_steps = 1000 # Tidstrin per sample
    eeg_channels = 64
    emg_channels = 1
    imu_channels = 6
    num_movements = 5 # Antallet of binære bevægelsesklasser
    
    # Simulated single-sample raw data (replace these with actual recordings)
    eeg_data = np.random.rand(eeg_channels, time_steps)
    emg_data = np.random.rand(emg_channels, time_steps)
    imu_data = np.random.rand(imu_channels, time_steps)
    
    # Simulated binary movement labels (for single sample training)
    movement_labels = np.random.randint(0, 2, (num_movements,))
    
    # Initialize and train the detector for single-sample demonstration
    detector = MovementIntentionDetector(sampling_rate=500,
                                         time_steps=time_steps,
                                         eeg_channels=eeg_channels,
                                         emg_channels=emg_channels,
                                         imu_channels=imu_channels)
    
    print("Træner modellen på et single sample...")
    training_metrics = detector.train(eeg_data, emg_data, imu_data, movement_labels, num_epochs=50)
    print("\nTræningsmetrikker:")
    for key, value in training_metrics.items():
        print(f"{key}: {value:.4f} sekunder")
    
    print("\nUdfører Inferens på en single sample...")
    binary_signals, inference_time = detector.predict_movement_intention(eeg_data, emg_data, imu_data)
    print(f"Inferenstid: {inference_time:.4f} sekunder")
    print(f"Forudsagte Binære Bevægelsessignaler: {binary_signals}")
    detector.timing_metrics.print_summary()
    
    # -----------------------------
    # K-fold Cross-validation Demo
    # -----------------------------
    
    # Simulate a dataset with multiple samples.
    num_samples = 50
    # Data shape: (samples, channels, time)
    X_eeg = np.random.rand(num_samples, eeg_channels, time_steps)
    X_emg = np.random.rand(num_samples, emg_channels, time_steps)
    X_imu = np.random.rand(num_samples, imu_channels, time_steps)
    # Simulated binary labels: shape (samples, num_movements)
    y = np.random.randint(0, 2, (num_samples, num_movements))
    
    print("\nRunning K-fold Cross-Validation...")
    avg_acc, avg_auc, all_cm = run_kfold_cv(X_eeg, X_emg, X_imu, y, num_folds=5, num_epochs=30)
    
    # Plot ROC curves using the predictions from the last fold.
    # Using last fold's test samples:
    # For demonstration, get predictions on last fold test set again.
    test_sample_indices = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_eeg))[-1][1]
    y_test = y[test_sample_indices]
    y_scores = []
    for i in test_sample_indices:
        # Using the same detector from the last fold simulation
        detector = MovementIntentionDetector(sampling_rate=500,
                                             time_steps=time_steps,
                                             eeg_channels=eeg_channels,
                                             emg_channels=emg_channels,
                                             imu_channels=imu_channels)
        # For simulation, train on a single sample (mean over training)
        detector.train(np.mean(X_eeg, axis=0), 
                       np.mean(X_emg, axis=0), 
                       np.mean(X_imu, axis=0), 
                       np.mean(y, axis=0), num_epochs=30)
        detector.model.eval()
        with torch.no_grad():
            eeg_tensor, emg_tensor, imu_tensor = detector.preprocess_signals(X_eeg[i], X_emg[i], X_imu[i])
            logits = detector.model(eeg_tensor, emg_tensor, imu_tensor)
            probs = torch.sigmoid(logits).numpy()
        y_scores.append(probs)
    y_scores = np.vstack(y_scores)
    
    print("\nPlotting ROC Curves for the last fold test samples:")
    _ = plot_roc_curves(y_test, y_scores, num_movements)

if __name__ == "__main__":
    main()
