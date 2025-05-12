# AI Data Center Load Modeling Project

This project focuses on modeling and predicting data center power consumption based on the Google Cluster Dataset. It employs multiple deep learning approaches including LSTM networks, Variational Autoencoders (VAE), and diffusion models to analyze, forecast, and generate synthetic power utilization patterns.

## Project Overview

The project aims to study the relationship between computational workloads and power consumption in data centers. It processes Google's cluster and power usage data, trains various models for time-series prediction and generation, and provides analysis tools for evaluating the results.

## Data

The project uses the Google Cluster Data 2019 dataset, focusing on:
- Machine utilization data 
- Power consumption data from PDUs (Power Distribution Units)
- Data is processed from specific data center cells (e.g., 'a') and PDUs (e.g., 'pdu6', 'pdu7')

## Files and Components

### Data Processing

- **process_google_data.py**: Fetches and processes raw data from Google BigQuery, creates time-aligned datasets of machine workloads and power consumption, and outputs processed data in Feather format.
  - Selects specific PDUs and machine samples
  - Aggregates data to 15-minute intervals
  - Creates merged datasets that correlate workload and power measurements

### Model Training

- **train_lstm_model.py**: Implements a PyTorch LSTM model for time-series prediction of power utilization.
  - Creates sequence data from the processed datasets
  - Trains an LSTM model to predict future power consumption
  - Saves trained models and evaluation metrics

- **train_vae_model.py**: Implements a Variational Autoencoder for time-series modeling.
  - Uses LSTM-based encoder and decoder architectures
  - Learns latent representations of power and workload patterns
  - Can generate synthetic time-series data

- **tdiffusion_time_series.ipynb**: Implements and trains diffusion models for time-series data.
  - Uses denoising diffusion probabilistic models (DDPM) to learn power consumption patterns
  - Implements forward and reverse diffusion processes for time-series data
  - Performs hyperparameter optimization across different settings:
    - Window sizes (64, 96)
    - Learning rates (0.0005, 0.001)
    - Batch sizes
    - Latent dimensions
  - Tracks training progress with loss metrics
  - Includes early stopping mechanisms for efficient training
  - Generates synthetic power consumption patterns by sampling from the learned distribution

### Analysis and Visualization

- **analysis.ipynb**: Jupyter notebook for exploratory data analysis of the processed data.
  - Visualizes workload and power consumption patterns
  - Analyzes correlations between CPU usage and power

- **diff_analysis.ipynb**: Comprehensive analysis of the diffusion models' performance.
  - Evaluates the quality of generated samples against real data
  - Compares different model configurations and hyperparameters
  - Analyzes statistical properties of generated sequences
  - Visualizes the step-by-step denoising process
  - Includes temporal consistency checks for generated time-series
  - Compares diffusion models against VAE and LSTM approaches

- **ai_data_center_load_modeling.ipynb**: Main notebook for comprehensive modeling and analysis.
  - Integrates findings from all modeling approaches
  - Provides comparative analysis between different model types
  - Visualizes predictions and generations from all models

### Trained Model Outputs

- **pytorch_lstm_model_output/**: Directory containing trained LSTM models and outputs
- **vae_model_output/**: Directory containing trained VAE models and outputs
- **diffusion_model_output/**: Directory containing trained diffusion models and outputs
  - Includes model checkpoints at different training stages
  - Contains configuration files for successful model parameters
  - Stores generated sample outputs from the diffusion process

### Data Files

- **merged_data_cell_a_pdu6_pdu7_approx100machines_30d.feather**: Processed dataset containing ~100 machines' worth of data over 30 days
- **X_train.npy** and **X_val.npy**: Preprocessed numpy arrays used for model training

## Project Outcomes

This project demonstrates:
1. How to process and align machine workload data with power consumption data
2. Implementation of various deep learning approaches for time-series modeling
3. Techniques for generating synthetic time-series data that preserves temporal patterns
   - LSTM-based prediction methods
   - VAE-based generative modeling
   - Diffusion-based probabilistic modeling for high-quality time-series generation
4. Analysis methods for evaluating model performance in terms of predictive accuracy and generation quality
5. Comparative analysis showing the strengths and weaknesses of each approach for data center power modeling
