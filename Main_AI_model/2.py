#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omfattende Multimodal Bevægelsesintentions Detektionssystem
"""

# ============================================================
# 1. CORE IMPORTS
# ============================================================
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal

# Machine Learning Imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    precision_score, 
    recall_score,
    confusion_matrix, 
    roc_curve, 
    auc, 
    f1_score, 
    classification_report,
    precision_recall_curve
)

# Visualiserings Imports
import matplotlib.pyplot as plt
import seaborn as sns

# Enum for Bevægelsesklassifikation
from enum import Enum, auto
import serial

# ============================================================
# 2. BEVÆGELSESKLASSIFIKATION
# ============================================================
class MovementType(Enum):
    """
    Prædefinerede bevægelsestyper med multi-class klassifikation
    """
    REST = 0b0000            # 0
    FLEXION = 0b0001         # 1
    EXTENSION = 0b0010       # 2
    RAISE = 0b0011           # 3
    UNRAISE = 0b0100         # 4
    SUPINATION = 0b0101      # 5
    PRONATION = 0b0110       # 6
    
# ============================================================
# 3. MODEL KONFIGURATION
# ============================================================
class ModelConfig:
    """
    Konfigurationsindstillinger for bevægelsesintentions-modellen
    """
    # Signal Parametre
    SAMPLING_RATE = 256  # Hz
    TIME_STEPS = 1000
    
    # Bereitschaftspotential Vindue
    BP_START = 0.5  # sekunder før bevægelse
    BP_END = 2.0    # sekunder før bevægelse
    
    # Model Hyperparametre
    D_MODEL = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3
    DROPOUT = 0.3
    
    # Kanal Konfiguration
    EEG_CHANNELS = 3  # C3, C4, Cz
    EMG_CHANNELS = 1
    IMU_CHANNELS = 6
    
    # Trænings Parametre
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # Inferens Parametre
    PREDICTION_THRESHOLD = 0.5

# ============================================================
# 4. DATA LOADER
# ============================================================
class MovementDataLoader:
    """
    Klasse til indlæsning og generering af bevægelses-data
    """
    @staticmethod
    def load_movement_data(
        eeg_path, 
        emg_path, 
        imu_path, 
        labels_path
    ):
        """
        Indlæs multimodal bevægelsesdata fra filer
        """
        try:
            # Indlæs data
            eeg_data = pd.read_csv(eeg_path).values
            emg_data = pd.read_csv(emg_path).values
            imu_data = pd.read_csv(imu_path).values
            labels = pd.read_csv(labels_path)['movement_type'].values
            
            # Encode labels direkte
            encoded_labels = np.array([MovementType[label].value for label in labels])
            
            return (
                eeg_data.reshape(-1, ModelConfig.EEG_CHANNELS, ModelConfig.TIME_STEPS),
                emg_data.reshape(-1, ModelConfig.EMG_CHANNELS, ModelConfig.TIME_STEPS),
                imu_data.reshape(-1, ModelConfig.IMU_CHANNELS, ModelConfig.TIME_STEPS),
                encoded_labels
            )
        except Exception as e:
            print(f"Datafejl ved indlæsning: {e}")
            return None

    @staticmethod
    def generate_synthetic_data(
        num_samples=1000, 
        movement_distribution=None
    ):
        """
        Generer syntetisk multimodal bevægelsesdata
        """
        # Standard bevægelsesfordeling
        if movement_distribution is None:
            movement_distribution = {
                MovementType.REST: 0.3,
                MovementType.EXTENSION: 0.15,
                MovementType.FLEXION: 0.15,
                MovementType.RAISE: 0.1,
                MovementType.UNRAISE: 0.1,
                MovementType.SUPINATION: 0.1,
                MovementType.PRONATION: 0.1
            }
        
        # Generer bevægelsesmærker
        movements = list(movement_distribution.keys())
        labels = np.random.choice(
            movements, 
            size=num_samples, 
            p=list(movement_distribution.values())
        )
        
        # Encode labels direkte ved brug af enum-værdier
        encoded_labels = np.array([movement.value for movement in labels])
        
        # Generer syntetiske signaldata med mere realistiske karakteristika
        def generate_realistic_signal(num_samples, num_channels, time_steps):
            base = np.random.randn(num_samples, num_channels, time_steps)
            # Tilføj struktureret variation
            for i in range(num_samples):
                # Simuler bevægelses-lignende mønstre
                movement_intensity = encoded_labels[i] != MovementType.REST.value
                base[i] += movement_intensity * np.sin(np.linspace(0, 10, time_steps))
            return base
        
        X_eeg = generate_realistic_signal(
            num_samples, 
            ModelConfig.EEG_CHANNELS, 
            ModelConfig.TIME_STEPS
        )
        X_emg = generate_realistic_signal(
            num_samples, 
            ModelConfig.EMG_CHANNELS, 
            ModelConfig.TIME_STEPS
        )
        X_imu = generate_realistic_signal(
            num_samples, 
            ModelConfig.IMU_CHANNELS, 
            ModelConfig.TIME_STEPS
        )
        
        return X_eeg, X_emg, X_imu, encoded_labels

# ============================================================
# 5. SIGNAL PREPROCESSING
# ============================================================
class SignalPreprocessor:
    """
    Klasse til forbehandling af signaldata
    """
    @staticmethod
    def bandpass_filter(
        signal_data, 
        sampling_rate, 
        lowcut=1, 
        highcut=40
    ):
        """
        Anvend båndpasfilter på input-signal
        
        Args:
            signal_data (np.array): Input signaldata
            sampling_rate (int): Samplingshastighed
            lowcut (float): Nedre grænsefrekvens
            highcut (float): Øvre grænsefrekvens
        
        Returns:
            np.array: Filtreret signaldata
        """
        nyquist = 0.5 * sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(N=6, Wn=[low, high], btype='band')
        
        if signal_data.ndim == 2:
            return np.array([signal.filtfilt(b, a, sig) for sig in signal_data])
        return signal.filtfilt(b, a, signal_data)
    
    @staticmethod
    def normalize(signal_data):
        """
        Z-score normalisering af signal
        
        Args:
            signal_data (np.array): Input signaldata
        
        Returns:
            np.array: Normaliseret signaldata
        """
        scaler = StandardScaler()
        if signal_data.ndim == 3:
            normalized_data = np.zeros_like(signal_data)
            for i in range(signal_data.shape[0]):
                normalized_data[i] = scaler.fit_transform(signal_data[i].T).T
            return normalized_data
        return scaler.fit_transform(signal_data.T).T

    @staticmethod
    def extract_bereitschaftspotential(
        eeg_data, 
        sampling_rate=ModelConfig.SAMPLING_RATE, 
        bp_start=ModelConfig.BP_START, 
        bp_end=ModelConfig.BP_END
    ):
        """
        Udtræk Bereitschaftspotential segment fra EEG-data
        
        Args:
            eeg_data (np.array): EEG signaldata
            sampling_rate (int): Samplingshastighed
            bp_start (float): Start tid før bevægelse
            bp_end (float): Slut tid før bevægelse
        
        Returns:
            np.array: Bereitschaftspotential segment
        """
        start_sample = int(bp_start * sampling_rate)
        end_sample = int(bp_end * sampling_rate)
        
        return eeg_data[:, :, -end_sample:-start_sample]

# ============================================================
# 6. MULTIMODAL TRANSFORMER MODEL
# ============================================================
class BereitschaftspotentialTransformer(nn.Module):
    """
    Transformer-baseret model til bevægelsesintentions-klassifikation
    """
    def __init__(
        self, 
        eeg_channels=ModelConfig.EEG_CHANNELS,
        emg_channels=ModelConfig.EMG_CHANNELS,
        imu_channels=ModelConfig.IMU_CHANNELS,
        d_model=ModelConfig.D_MODEL,
        num_heads=ModelConfig.NUM_HEADS,
        num_layers=ModelConfig.NUM_LAYERS,
        dropout=ModelConfig.DROPOUT
    ):
        super().__init__()
        
        # Modalitets-specifikke konvolutionslag
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(eeg_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.emg_conv = nn.Sequential(
            nn.Conv1d(emg_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.imu_conv = nn.Sequential(
            nn.Conv1d(imu_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Projektionslag
        self.eeg_proj = nn.Linear(64, d_model)
        self.emg_proj = nn.Linear(32, d_model)
        self.imu_proj = nn.Linear(32, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Klassifikationshoved
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, len(MovementType)),  # Dynamisk antal klasser
            nn.Softmax(dim=1)  # Softmax for multi-class klassifikation
        )
    
    def forward(self, eeg, emg, imu):
        """
        Fremadrettet gennemløb for modellen
        
        Args:
            eeg (torch.Tensor): EEG input
            emg (torch.Tensor): EMG input
            imu (torch.Tensor): IMU input
        
        Returns:
            torch.Tensor: Klassifikationssandsynligheder
        """
        # Modalitets-specifik feature-ekstration
        eeg_feat = self.eeg_conv(eeg)
        eeg_feat = eeg_feat.permute(0, 2, 1)
        eeg_feat = self.eeg_proj(eeg_feat)
        
        emg_feat = self.emg_conv(emg)
        emg_feat = emg_feat.permute(0, 2, 1)
        emg_feat = self.emg_proj(emg_feat)
        
        imu_feat = self.imu_conv(imu)
        imu_feat = imu_feat.permute(0, 2, 1)
        imu_feat = self.imu_proj(imu_feat)
        
        # Kombiner modaliteter
        combined = torch.cat([eeg_feat, emg_feat, imu_feat], dim=1)
        
        # Transformer behandling
        transformer_out = self.transformer(combined)
        
        # Gennemsnitlig pooling
        pooled = transformer_out.mean(dim=1)
        
        # Klassifikation
        return self.classifier(pooled)

# ============================================================
# 7. PERFORMANCE EVALUATOR
# ============================================================
class ModelPerformanceEvaluator:
    """
    Klasse til evaluering af modelperformance
    """
    def __init__(self, model, device):
        """
        Initialisér performance evaluator
        
        Args:
            model (nn.Module): Den trænede model
            device (torch.device): Beregningsenhed
        """
        self.model = model
        self.device = device

    def evaluate_performance(
        self, 
        X_eeg, X_emg, X_imu, y_true, 
        threshold=ModelConfig.PREDICTION_THRESHOLD
    ):
        """
        Omfattende evaluering af modelperformance
        
        Args:
            X_eeg (np.array): EEG input data
            X_emg (np.array): EMG input data
            X_imu (np.array): IMU input data
            y_true (np.array): Sande labels
            threshold (float): Klassifikationstærskel
        
        Returns:
            dict: Performance metrikker
        """
        # Forbered tensorer
        eeg_tensor = torch.FloatTensor(X_eeg).to(self.device)
        emg_tensor = torch.FloatTensor(X_emg).to(self.device)
        imu_tensor = torch.FloatTensor(X_imu).to(self.device)
        
        # Måling af inferenstid
        start_time = time.time()
        
        # Forudsigelse
        with torch.no_grad():
            probabilities = self.model(eeg_tensor, emg_tensor, imu_tensor)
        
        # Beregn inferenstid
        inference_time = time.time() - start_time
        
        # Konverter til numpy
        y_pred_prob = probabilities.cpu().numpy()
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Performance metrikker
        metrics = {
            'inference_time': inference_time,
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        # ROC Curve og AUC (tilpasset multi-class)
        # Bemærk: Dette er en forenklet version
        fpr, tpr = None, None
        metrics['roc_auc'] = 0  # Kan forbedres med one-vs-rest
        
        # Precision-Recall Curve
        precision, recall = None, None
        metrics['pr_auc'] = 0  # Kan forbedres med one-vs-rest
        
        return metrics, y_pred_prob, fpr, tpr, precision, recall

    def visualize_performance(self, metrics, y_true, y_pred_prob, fpr, tpr, precision, recall):
        """
        Skab omfattende visualisering af modelperformance
        
        Args:
            metrics (dict): Performance metrikker
            y_true (np.array): Sande labels
            y_pred_prob (np.array): Forudsagte sandsynligheder
            fpr (np.array): False positive rate
            tpr (np.array): True positive rate
            precision (np.array): Precision værdier
            recall (np.array): Recall værdier
        """
        plt.figure(figsize=(15, 5))
        
        # 1. Konfusionsmatrix
        plt.subplot(131)
        sns.heatmap(
            metrics['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[mt.name for mt in MovementType],
            yticklabels=[mt.name for mt in MovementType]
        )
        plt.title('Konfusionsmatrix')
        plt.xlabel('Forudsagt')
        plt.ylabel('Faktisk')
        
        # 2. Performance rapport
        plt.subplot(132)
        plt.text(0.1, 0.5, metrics['classification_report'], 
                 fontsize=10, family='monospace')
        plt.title('Klassifikationsrapport')
        plt.axis('off')
        
        # 3. Performance oversigt
        plt.subplot(133)
        performance_data = [
            f"Inferenstid: {metrics['inference_time']:.4f} sek",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
            f"F1-Score: {metrics['f1_score']:.4f}"
        ]
        plt.text(0.1, 0.5, "\n".join(performance_data), 
                 fontsize=10, family='monospace')
        plt.title('Performance Oversigt')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def generate_detailed_report(self, metrics):
        """
        Generer omfattende performance rapport
        
        Args:
            metrics (dict): Performance metrikker
        """
        print("\n--- Model Performance Rapport ---")
        print(f"Inferenstid: {metrics['inference_time']:.4f} sekunder")
        print("\nKlassifikationsrapport:")
        print(metrics['classification_report'])
        
        print("\nKonfusionsmatrix:")
        print(metrics['confusion_matrix'])
        
        print("\nHovedmetrikker:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")

# ============================================================
# 8. BEVÆGELSESINTENTIONS DETEKTOR
# ============================================================
class MovementIntentionDetector:
    """
    Klasse til detektion og klassifikation af bevægelsesintentioner
    """
    def __init__(
        self, 
        sampling_rate=ModelConfig.SAMPLING_RATE,
        eeg_channels=ModelConfig.EEG_CHANNELS,
        serial_port='/dev/ttyUSB0',  # Juster til din Arduino port
        baud_rate=57600
    ):
        """
        Initialisér bevægelsesintentions detektor
        
        Args:
            sampling_rate (int): Signalens samplingshastighed
            eeg_channels (int): Antal EEG kanaler
            serial_port (str): Serial port til Arduino
            baud_rate (int): Kommunikationshastighed
        """
        self.sampling_rate = sampling_rate
        self.eeg_channels = eeg_channels
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.performance_metrics = None
        
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            print(f"Forbundet til Arduino på {serial_port}")
        except serial.SerialException as e:
            print(f"Kunne ikke forbinde til serial port: {e}")
            self.ser = None

    def prepare_data(self, eeg_data, emg_data, imu_data):
        """
        Forbered input data til modellen
        
        Args:
            eeg_data (np.array): EEG input data
            emg_data (np.array): EMG input data
            imu_data (np.array): IMU input data
        
        Returns:
            tuple: Forberedte tensorer
        """
        eeg_tensor = torch.FloatTensor(eeg_data).to(self.device)
        emg_tensor = torch.FloatTensor(emg_data).to(self.device)
        imu_tensor = torch.FloatTensor(imu_data).to(self.device)

        return eeg_tensor, emg_tensor, imu_tensor

    def predict_and_send(self, eeg_data, emg_data, imu_data):
        """
        Forudsig bevægelsestype og send via serial
        
        Args:
            eeg_data (np.array): EEG input data
            emg_data (np.array): EMG input data
            imu_data (np.array): IMU input data
        
        Returns:
            int: Forudsagt bevægelsestype
        """
        if self.model is None:
            raise ValueError("Model ikke trænet. Kald train() først.")
    
        self.model.eval()
        with torch.no_grad():
            eeg_tensor, emg_tensor, imu_tensor = self.prepare_data(
                eeg_data, emg_data, imu_data
            )
            probabilities = self.model(eeg_tensor, emg_tensor, imu_tensor)
        
        # Find bevægelsestype med højeste sandsynlighed
        predicted_movement = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
    
        # Send bevægelsestype via serial
        if self.ser:
            try:
                self.ser.write(f"{predicted_movement}\n".encode())
                print(f"Sendt bevægelsestype: {MovementType(predicted_movement).name}")
            except Exception as e:
                print(f"Fejl ved serial afsendelse: {e}")
    
        return predicted_movement

    def save_model(self, path='movement_intention_model.pth'):
        """
        Gem den trænede model
        
        Args:
            path (str): Filsti til model-gemning
        """
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'movement_types': [mt.name for mt in MovementType],
                'model_config': {
                    'eeg_channels': ModelConfig.EEG_CHANNELS,
                    'emg_channels': ModelConfig.EMG_CHANNELS,
                    'imu_channels': ModelConfig.IMU_CHANNELS,
                    'd_model': ModelConfig.D_MODEL,
                    'num_heads': ModelConfig.NUM_HEADS,
                    'num_layers': ModelConfig.NUM_LAYERS,
                    'dropout': ModelConfig.DROPOUT
                }
            }, path)
            print(f"Model gemt i {path}")
        else:
            print("Ingen model at gemme")

    def load_model(self, path='movement_intention_model.pth'):
        """
        Indlæs en gemt model
        
        Args:
            path (str): Filsti til model-indlæsning
        """
        try:
            checkpoint = torch.load(path)
            
            # Genopret model med samme konfiguration
            self.model = BereitschaftspotentialTransformer(
                eeg_channels=checkpoint['model_config']['eeg_channels'],
                emg_channels=checkpoint['model_config']['emg_channels'],
                imu_channels=checkpoint['model_config']['imu_channels'],
                d_model=checkpoint['model_config']['d_model'],
                num_heads=checkpoint['model_config']['num_heads'],
                num_layers=checkpoint['model_config']['num_layers'],
                dropout=checkpoint['model_config']['dropout']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"Model indlæst fra {path}")
            print("Gemte bevægelsestyper:", checkpoint['movement_types'])
        except Exception as e:
            print(f"Fejl ved indlæsning af model: {e}")

    def train(
        self, 
        X_eeg, 
        X_emg, 
        X_imu, 
        y, 
        epochs=ModelConfig.EPOCHS
    ):
            """
            Træn bevægelsesintentions-model
            
            Args:
                X_eeg (np.array): EEG input data
                X_emg (np.array): EMG input data
                X_imu (np.array): IMU input data
                y (np.array): Bevægelseslabels
                epochs (int): Antal træningsepoch'er
            
            Returns:
                dict: Performance metrikker
            """
            # Del data i trænings- og valideringssæt
            X_eeg_train, X_eeg_val, X_emg_train, X_emg_val, \
            X_imu_train, X_imu_val, y_train, y_val = train_test_split(
                X_eeg, X_emg, X_imu, y, test_size=0.2, random_state=42
            )
            
            # Initialisér model
            self.model = BereitschaftspotentialTransformer().to(self.device)
            
            # Tab-funktion og optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=ModelConfig.LEARNING_RATE
            )
            
            # Træningsloop
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                # Batch-træning
                for i in range(0, len(X_eeg_train), ModelConfig.BATCH_SIZE):
                    batch_eeg = X_eeg_train[i:i+ModelConfig.BATCH_SIZE]
                    batch_emg = X_emg_train[i:i+ModelConfig.BATCH_SIZE]
                    batch_imu = X_imu_train[i:i+ModelConfig.BATCH_SIZE]
                    batch_labels = y_train[i:i+ModelConfig.BATCH_SIZE]
                    
                    # Forbered batch
                    eeg_tensor, emg_tensor, imu_tensor = self.prepare_data(
                        batch_eeg, batch_emg, batch_imu
                    )
                    labels = torch.LongTensor(batch_labels).to(self.device)
                    
                    # Nulstil gradienter
                    optimizer.zero_grad()
                    
                    # Fremadrettet gennemløb
                    outputs = self.model(eeg_tensor, emg_tensor, imu_tensor)
                    loss = criterion(outputs, labels)
                    
                    # Bagudrettet gennemløb
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Validering
                self.model.eval()
                with torch.no_grad():
                    eeg_val, emg_val, imu_val = self.prepare_data(
                        X_eeg_val, X_emg_val, X_imu_val
                    )
                    val_labels = torch.LongTensor(y_val).to(self.device)
                    val_outputs = self.model(eeg_val, emg_val, imu_val)
                    val_loss = criterion(val_outputs, val_labels)
               
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Trænings-tab: {total_loss/len(X_eeg_train):.4f}, "
                      f"Validerings-tab: {val_loss.item():.4f}")
            
            # Evaluer performance
            evaluator = ModelPerformanceEvaluator(self.model, self.device)
            self.performance_metrics, y_pred_prob, fpr, tpr, precision, recall = evaluator.evaluate_performance(
                X_eeg_val, X_emg_val, X_imu_val, y_val
            )
            
            # Visualisér og rapportér
            evaluator.generate_detailed_report(self.performance_metrics)
            evaluator.visualize_performance(
                self.performance_metrics, 
                y_val, 
                y_pred_prob, 
                fpr, 
                tpr, 
                precision, 
                recall
            )
            
            return self.performance_metrics
    
    def cross_validate(
            self, 
            X_eeg, 
            X_emg, 
            X_imu, 
            y, 
            n_splits=5,
            epochs=ModelConfig.EPOCHS
        ):
            """
            Udfør k-fold cross-validation
            
            Args:
                X_eeg (np.array): EEG input data
                X_emg (np.array): EMG input data
                X_imu (np.array): IMU input data
                y (np.array): Bevægelseslabels
                n_splits (int): Antal fold i cross-validation
                epochs (int): Antal træningsepoch'er
            
            Returns:
                tuple: Cross-validation metrikker og fold-resultater
            """
            # Initialisér K-Fold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Sporing af metrikker
            cv_metrics = {
                'precision': [],
                'recall': [],
                'f1_score': [],
                'roc_auc': [],
                'inferenstider': []
            }
            
            # Fold-vise resultater
            fold_resultater = []
            
            # Cross-validation loop
            for fold, (train_index, val_index) in enumerate(kf.split(X_eeg), 1):
                print(f"\n--- Fold {fold}/{n_splits} ---")
                
                # Del data
                X_eeg_train, X_eeg_val = X_eeg[train_index], X_eeg[val_index]
                X_emg_train, X_emg_val = X_emg[train_index], X_emg[val_index]
                X_imu_train, X_imu_val = X_imu[train_index], X_imu[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                # Træn model for dette fold
                self.model = BereitschaftspotentialTransformer().to(self.device)
                
                # Træn model
                self.train(X_eeg_train, X_emg_train, X_imu_train, y_train, epochs)
                
                # Evaluer performance
                evaluator = ModelPerformanceEvaluator(self.model, self.device)
                metrics, y_pred_prob, fpr, tpr, precision, recall = evaluator.evaluate_performance(
                    X_eeg_val, X_emg_val, X_imu_val, y_val
                )
                
                # Gem fold-resultater
                fold_resultater.append({
                    'metrikker': metrics,
                    'sande_labels': y_val,
                    'forudsagte_sandsynligheder': y_pred_prob
                })
                
                # Aggregér metrikker
                cv_metrics['precision'].append(metrics['precision'])
                cv_metrics['recall'].append(metrics['recall'])
                cv_metrics['f1_score'].append(metrics['f1_score'])
                cv_metrics['roc_auc'].append(metrics.get('roc_auc', 0))
                cv_metrics['inferenstider'].append(metrics['inference_time'])
            
            # Udskriv cross-validation oversigt
            self._print_cv_summary(cv_metrics)
            
            return cv_metrics, fold_resultater
    
    def _print_cv_summary(self, cv_metrics):
            """
            Udskriv oversigt over cross-validation resultater
            
            Args:
                cv_metrics (dict): Cross-validation metrikker
            """
            print("\n--- Cross-Validation Oversigt ---")
            for metrik, værdier in cv_metrics.items():
                print(f"{metrik}:")
                print(f"  Gennemsnit: {np.mean(værdier):.4f}")
                print(f"  Standardafvigelse: {np.std(værdier):.4f}")
    
    def predict(self, eeg_data, emg_data, imu_data):
            """
            Forudsig bevægelsesintention
            
            Args:
                eeg_data (np.array): EEG input data
                emg_data (np.array): EMG input data
                imu_data (np.array): IMU input data
            
            Returns:
                np.array: Sandsynligheder for bevægelsestyper
            """
            if self.model is None:
                raise ValueError("Model ikke trænet. Kald train() først.")
            
            self.model.eval()
            with torch.no_grad():
                eeg_tensor, emg_tensor, imu_tensor = self.prepare_data(
                    eeg_data, emg_data, imu_data
                )
                probabilities = self.model(eeg_tensor, emg_tensor, imu_tensor)
            
            return probabilities.cpu().numpy()[0]
# ============================================================
# 9. HOVEDUDFØRELSE
# ============================================================
def main():
    """
    Hovedfunktion til demonstration og test af bevægelsesintentions-systemet
    """
    # Vælg dataindlæsningsmetode
    try:
        # Mulighed 1: Generer syntetiske data
        X_eeg, X_emg, X_imu, y = MovementDataLoader.generate_synthetic_data(
            num_samples=1000,
            movement_distribution={
                MovementType.REST: 0.3,
                MovementType.EXTENSION: 0.15,
                MovementType.FLEXION: 0.15,
                MovementType.RAISE: 0.1,
                MovementType.UNRAISE: 0.1,
                MovementType.SUPINATION: 0.1,
                MovementType.PRONATION: 0.1
            }
        )
        
        # Mulighed 2: Indlæs rigtige data (kommenteret ud)
        # X_eeg, X_emg, X_imu, y = MovementDataLoader.load_movement_data(
        #     eeg_path='eeg_data.csv',
        #     emg_path='emg_data.csv',
        #     imu_path='imu_data.csv',
        #     labels_path='labels.csv'
        # )
        
        # Initialisér detektor med serial kommunikation
        detector = MovementIntentionDetector(
            serial_port='/dev/ttyUSB0',  # Juster til din Arduino port
            baud_rate=57600
        )
        
        # Vælg træningsmetode
        print("Vælg træningsmetode:")
        print("1. Enkelt træning")
        print("2. Cross-validation")
        valg = input("Indtast dit valg (1/2): ")
        
        if valg == '1':
            # Træn model
            performance_metrics = detector.train(X_eeg, X_emg, X_imu, y)
            
            # Gem model
            detector.save_model('movement_intention_model.pth')
        
        elif valg == '2':
            # Udfør cross-validation
            cv_metrics, fold_resultater = detector.cross_validate(
                X_eeg, X_emg, X_imu, y
            )
        
        else:
            print("Ugyldigt valg. Afslutter.")
            return
        
        # Kontinuerlig forudsigelse og afsendelse
        try:
            while True:
                # Indlæs nye sensordata (erstat med din faktiske dataindsamling)
                test_eeg = X_eeg[np.random.randint(len(X_eeg))]
                test_emg = X_emg[np.random.randint(len(X_emg))]
                test_imu = X_imu[np.random.randint(len(X_imu))]
                
                # Forudsig og send bevægelsestype
                movement_intention = detector.predict_and_send(
                    test_eeg.reshape(1, *test_eeg.shape), 
                    test_emg.reshape(1, *test_emg.shape), 
                    test_imu.reshape(1, *test_imu.shape)
                )
                
                # Lille pause mellem forudsigelser
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nStopper bevægelsesintentions-detektion")
            if detector.ser:
                detector.ser.close()
    
    except Exception as e:
        print(f"En fejl opstod: {e}")

# Sikrer at scriptet kun køres direkte
if __name__ == "__main__":
    main()

# Arduino-kode til reference
"""
void loop() {
  if (Serial.available()) {
    int movement_type = Serial.parseInt();
    
    // Handling baseret på bevægelsestype
    switch(movement_type) {
      case 0:  // REST
        digitalWrite(LED_BUILTIN, LOW);
        break;
      case 1:  // EXTENSION
        digitalWrite(LED_BUILTIN, HIGH);
        break;
      case 2:  // FLEXION
        // Specifikke aktioner
        break;
      case 3:  // RAISE
        // Specifikke aktioner
        break;
      case 4:  // UNRAISE
        // Specifikke aktioner
        break;
      case 5:  // SUPINATION
        // Specifikke aktioner
        break;
      case 6:  // PRONATION
        // Specifikke aktioner
        break;
    }
  }
}
"""
