local SEED = 23;
local CUDA = 0;
local READER = "lyrics-gan";
local PREDICTOR = 'dialog-gen';

local LATENT_DIM = 128;
local BATCH_SIZE = 32;
local TEMPERATURE = 1e-5;
local ACTIVATION = 'relu';

local NUM_EPOCHS = 30;
local PATIENCE = 5;
local SUMMARY_INTERVAL = 10;
local GEN_OPTIMIZER = "adam";
local DISC_OPTIMIZER = "adam";
local DISC_LEARNING_RATE = 0.001;
local GEN_LEARNING_RATE = 0.001;

{
  "random_seed": SEED,
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "dataset_reader": {
    "type": READER,
  },
  "train_data_path": "data/processed/train_data.pkl",
  "validation_data_path": "data/processed/valid_data.pkl",
  "model": {
    "type": "dialog_gan_latent",
    "latent_dim": LATENT_DIM,
    "generator": {
      "type": "dialog-generator",
      "latent_dim": LATENT_DIM,
      'activation': ACTIVATION,
      "initializer": [
        [".*", {"type": "normal", "mean": 0, "std": 0.02}],
      ]
    },
    "discriminator": {
      "type": "dialog-discriminator",
      "input_dim": 2*LATENT_DIM,
      "hidden_dim": LATENT_DIM,
      "initializer": [
        [".*", {"type": "normal", "mean": 0, "std": 0.02}],
      ]
    },
    "inference_temperature": TEMPERATURE,
  },
  "iterator": {
    "type": "homogeneous_batch",
    "batch_size" : BATCH_SIZE,
    "partition_key": "stage"
  },
  "trainer": {
    "type": 'callback',
    "num_epochs": NUM_EPOCHS,
    "cuda_device": CUDA,
    "optimizer": {
      "type": "gan",
      "generator_optimizer": {
        "type": GEN_OPTIMIZER,
        "lr": GEN_LEARNING_RATE
      },
      "discriminator_optimizer": {
        "type": DISC_OPTIMIZER,
        "lr": DISC_LEARNING_RATE
      }
    },
    "callbacks": [
      "gan-callback",
      "checkpoint",
      {"type": "track_metrics", "patience": PATIENCE, "validation_metric": "+_S_BLEU4F"},
      "log_to_tensorboard"
    ]
  }
}
