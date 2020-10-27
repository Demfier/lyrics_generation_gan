local SEED = 0;
local CUDA = 0;
local READER = "lyrics-gan";
local PREDICTOR = 'lyrics-gen';

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
  "vocabulary": {
    "directory_path": "models/lyrics_vae/vocabulary"
  },
  "train_data_path": "data/interim/lyrics/train_lyrics.tsv",
  "validation_data_path": "data/interim/lyrics/valid_lyrics.tsv",
  "model": {
    "type": "lyrics_gan",
    "encoder": {
      "_pretrained": {
        "archive_file": "models/lyrics_vae/model.tar.gz",
        "module_path": '_encoder',
        "freeze": true
      },
    },
    "decoder": {
      "_pretrained": {
        "archive_file": "models/lyrics_vae/model.tar.gz",
        "module_path": '_decoder',
        "freeze": true,
      },
    },
    "generator": {
      "type": "lyrics-generator",
      "latent_dim": LATENT_DIM,
      'activation': ACTIVATION,
      "initializer": [
        [".*", {"type": "normal", "mean": 0, "std": 0.02}],
      ]
    },
    "discriminator": {
      "type": "lyrics-discriminator",
      "input_dim": 2*LATENT_DIM,
      "hidden_dim": LATENT_DIM,
      "initializer": [
        [".*", {"type": "normal", "mean": 0, "std": 0.02}],
      ]
    },
    "inference_temperature": TEMPERATURE,
  },
  "iterator": {
    "type": "homogeneous_bucket",
    "batch_size" : BATCH_SIZE,
    "sorting_keys": [["source_tokens", "num_tokens"]],
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
      "validate",
      "generate_lyrics_samples",
      "log_to_tensorboard"
    ]
  }
}
