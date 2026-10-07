from ml_factory.training_scripts.training_script_ae import training as base_training, training_attack as base_training_attack, training_attack_contrastive as base_training_contrastive
import time
import datetime
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import matplotlib.pyplot as plt
from ml_factory import DATA_PROCESSED_DIR, DATA_RAW_DIR, TRAINING_LOGS_DIR, MODEL_DIRECTORY_DEV
from ml_factory.datasets import precompute_features_ae, PromptFeatureDataset
from ml_factory.datasets.sampler import RatioSampler, SplitSampler
import polars as pl
import numpy as np
from ml_factory.utils import embedding_text
from ml_factory.utils.scaler import StandardScaler
from ml_factory.models import BaseNormalAutoEncoder, Args
import asyncio
import hashlib

SEED = 3123

EMBEDDING_MODEL = "nomic-embed-text"
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
VALIDATION_RATIO = 0.15
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 200
ATTACK_RATIO = 0.2

def base_ae():
    processed_data_only_benign = DATA_PROCESSED_DIR / f'processed_benign_only_{EMBEDDING_MODEL}.npz'

    if not processed_data_only_benign.is_file():
        df = pl.scan_parquet(DATA_RAW_DIR / 'only_benign_prompts.parquet').with_columns(pl\
                .col('text')\
                .map_elements(lambda t: hashlib.sha256(t.encode()).hexdigest(), return_dtype=pl.Utf8).alias('id'))

        asyncio.run(precompute_features_ae(dataset_records=df,
                                           embed_fn=embedding_text,
                                           embedding_model_name=EMBEDDING_MODEL,
                                           out_path=processed_data_only_benign, batch_size=100, chunk_size=2, max_concurrent=6))

    dataset = PromptFeatureDataset(processed_data_only_benign)
    split_path = DATA_PROCESSED_DIR / f'processed_benign_only_seed_{SEED}_train_{TRAIN_RATIO}_test_{TEST_RATIO}.pt'

    if not split_path.is_file():
        splitter = SplitSampler(train_split=TRAIN_RATIO, test_split=TEST_RATIO, validation_split=VALIDATION_RATIO, seed=SEED)
        splitter.build_split(item_ids=dataset.item_ids, labels=dataset.labels)
        splitter.save_split(split_path)
    else:
        splitter = SplitSampler.load_file(split_path)

    std_scaler_path = DATA_PROCESSED_DIR / f'processed_benign_only_seed_{SEED}_train_{TRAIN_RATIO}_standard_sclaer.pt'

    train_ids = splitter.get_split('train')
    test_ids = splitter.get_split('test')
    validation_ids = splitter.get_split('validation')
    if not std_scaler_path.is_file():
        scaler = StandardScaler()
        train_positions = dataset.ids_to_positions(train_ids)
        scaler = scaler.fit(dataset.features[train_positions])
        dataset.features = scaler.transform(dataset.features)

        scaler.save(std_scaler_path)
    else:
        scaler = StandardScaler.load(std_scaler_path)
        dataset.features = scaler.transform(dataset.features)

    train_data = Subset(dataset, dataset.ids_to_positions(train_ids))

    test_data = Subset(dataset=dataset, indices=dataset.ids_to_positions(test_ids))

    validation_data = Subset(dataset=dataset, indices=dataset.ids_to_positions(validation_ids))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    args = Args()
    model = BaseNormalAutoEncoder(len(dataset[0][0]),args).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    datetime_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    time_name = time.time()
    best_model_settings, training_log, validation_log, test_log = base_training(training_loader=train_loader,
                                                                                val_loader=validation_loader,
                                                                                test_loader=test_loader,
                                                                                model=model,
                                                                                optimizer=optimizer,
                                                                                criterion=criterion,
                                                                                epoch=EPOCH,device=DEVICE)
    end_time = time.time() - time_name

    print(f"Training Took {end_time} seconds")
    training_log_dir = TRAINING_LOGS_DIR / f'{model.__class__.__name__} {datetime_string}'
    training_log_dir.mkdir(exist_ok=True)
    pl.DataFrame({'train_log': training_log, 'validation_log': validation_log}).write_csv(training_log_dir / f'{time_name}.csv')
    best_model = np.argmin(validation_log)

    plt.figure(figsize=(10, 10))
    plt.plot(range(0, len(training_log)), training_log, label='Training Loss')
    plt.plot(range(0, len(training_log)), validation_log, label='Validation Loss')
    plt.axvline(best_model, label='Best Model')
    plt.legend()
    plt.savefig(training_log_dir / f'{time_name}_LOSS_PLOT.png')
    plt.clf()

    with open(training_log_dir / f'{time_name}_details.txt', 'w') as f:
        f.write(f"""
MODEL NAME: {model.__class__.__name__}
Test Loss: {test_log}
Training Time Started: {datetime_string}
Time Taken: {end_time} seconds

# Parameters
EPOCHS: {EPOCH}
EMBEDDING_MODEL: {EMBEDDING_MODEL}
SEED: {SEED}
TRAIN_RATIO: {TRAIN_RATIO}
VALIDATION_RATIO: {VALIDATION_RATIO}
TEST_RATIO: {TEST_RATIO}
BATCH_SIZE: {BATCH_SIZE}

DATASET: {processed_data_only_benign}
        """)


    torch.save(best_model_settings, MODEL_DIRECTORY_DEV / f'{model.__class__.__name__}_{time_name}.pt')


def base_ae_attack():
    processed_data_only_benign = DATA_PROCESSED_DIR / f'processed_final_data_{EMBEDDING_MODEL}.npz'

    if not processed_data_only_benign.is_file():
        df = pl.scan_parquet(DATA_RAW_DIR / 'final_data.parquet').with_columns(pl\
                .col('text')\
                .map_elements(lambda t: hashlib.sha256(t.encode()).hexdigest(), return_dtype=pl.Utf8).alias('id'))

        asyncio.run(precompute_features_ae(dataset_records=df,
                                           embed_fn=embedding_text,
                                           embedding_model_name=EMBEDDING_MODEL,
                                           out_path=processed_data_only_benign, batch_size=100, chunk_size=2, max_concurrent=6))

    dataset = PromptFeatureDataset(processed_data_only_benign)
    split_path = DATA_PROCESSED_DIR / f'processed_final_data_seed_{SEED}_train_{TRAIN_RATIO}_test_{TEST_RATIO}.pt'

    if not split_path.is_file():
        splitter = SplitSampler(train_split=TRAIN_RATIO, test_split=TEST_RATIO, validation_split=VALIDATION_RATIO, seed=SEED)
        splitter.build_split(item_ids=dataset.item_ids, labels=dataset.labels)
        splitter.save_split(split_path)
    else:
        splitter = SplitSampler.load_file(split_path)

    std_scaler_path = DATA_PROCESSED_DIR / f'processed_final_data_seed_{SEED}_train_{TRAIN_RATIO}_standard_sclaer.pt'

    train_ids = splitter.get_split('train')
    test_ids = splitter.get_split('test')
    validation_ids = splitter.get_split('validation')
    if not std_scaler_path.is_file():
        scaler = StandardScaler()
        train_positions = dataset.ids_to_positions(train_ids)
        scaler = scaler.fit(dataset.features[train_positions])
        dataset.features = scaler.transform(dataset.features)

        scaler.save(std_scaler_path)
    else:
        scaler = StandardScaler.load(std_scaler_path)
        dataset.features = scaler.transform(dataset.features)

    train_positions = dataset.ids_to_positions(train_ids)
    train_data = Subset(dataset, train_positions)

    test_data = Subset(dataset=dataset, indices=dataset.ids_to_positions(test_ids))

    validation_data = Subset(dataset=dataset, indices=dataset.ids_to_positions(validation_ids))

    ratio_sampler = RatioSampler(dataset.labels[train_positions], attack_label=1, attack_ratio=ATTACK_RATIO, seed=SEED)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=ratio_sampler,num_workers=0)

    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)
    args = Args()
    model = BaseNormalAutoEncoder(len(dataset[0][0]),args).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    datetime_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    time_name = time.time()
    best_model_settings, (training_log, training_benign_log, training_attack_log), \
            (validation_log, validation_benign_log, validation_attack_log), \
            (test_log, test_benign_log, test_attack_log) = base_training_attack(training_loader=train_loader,
                                                                                val_loader=validation_loader,
                                                                                test_loader=test_loader,
                                                                                model=model,
                                                                                optimizer=optimizer,
                                                                                epoch=EPOCH,device=DEVICE)
    end_time = time.time() - time_name

    print(f"Training Took {end_time} seconds")
    training_log_dir = TRAINING_LOGS_DIR / f'{model.__class__.__name__} {datetime_string}'
    training_log_dir.mkdir(exist_ok=True)
    pl.DataFrame({'train_log': training_log,
                  'train_benign_log': training_benign_log,
                  'train_attack_log': training_attack_log,
                  'validation_log': validation_log,
                  'validation_benign_log': validation_benign_log,
                  'validation_attack_log': validation_attack_log
                  }).write_csv(training_log_dir / f'{time_name}.csv')
    best_model = np.argmin(validation_log)

    plt.figure(figsize=(10, 10))
    X_range = range(0, len(training_log))
    plt.plot(X_range, training_log, label='Training Loss')
    plt.plot(X_range, validation_log, label='Validation Loss')
    plt.plot(X_range, training_attack_log, label='Training Attack Loss')
    plt.plot(X_range, training_benign_log, label='Training Benign Loss')
    plt.plot(X_range, validation_attack_log, label='Validation Attack Loss')
    plt.plot(X_range, validation_benign_log, label='Validation Benign Loss')
    plt.axvline(best_model, label='Best Model')
    plt.legend()
    plt.savefig(training_log_dir / f'{time_name}_LOSS_PLOT.png')
    plt.clf()

    with open(training_log_dir / f'{time_name}_details.txt', 'w') as f:
        f.write(f"""
MODEL NAME: {model.__class__.__name__} (attack)
Test Loss: {test_log}
Test Attack Loss: {test_attack_log} (+should be higher)
Test Benign Loss: {test_benign_log}
Training Time Started: {datetime_string}
Time Taken: {end_time} seconds


# Parameters
EPOCHS: {EPOCH}
EMBEDDING_MODEL: {EMBEDDING_MODEL}
SEED: {SEED}
TRAIN_RATIO: {TRAIN_RATIO}
VALIDATION_RATIO: {VALIDATION_RATIO}
TEST_RATIO: {TEST_RATIO}
BATCH_SIZE: {BATCH_SIZE}
ATTACK_RATIO: {ATTACK_RATIO}

DATASET: {processed_data_only_benign}
        """)


    torch.save(best_model_settings, MODEL_DIRECTORY_DEV / f'{model.__class__.__name__}_attack_{time_name}.pt')



def base_ae_attack_contrastive():
    processed_data_only_benign = DATA_PROCESSED_DIR / f'processed_final_data_{EMBEDDING_MODEL}.npz'

    if not processed_data_only_benign.is_file():
        df = pl.scan_parquet(DATA_RAW_DIR / 'final_data.parquet').with_columns(pl\
                .col('text')\
                .map_elements(lambda t: hashlib.sha256(t.encode()).hexdigest(), return_dtype=pl.Utf8).alias('id'))

        asyncio.run(precompute_features_ae(dataset_records=df,
                                           embed_fn=embedding_text,
                                           embedding_model_name=EMBEDDING_MODEL,
                                           out_path=processed_data_only_benign, batch_size=100, chunk_size=2, max_concurrent=6))

    dataset = PromptFeatureDataset(processed_data_only_benign)
    split_path = DATA_PROCESSED_DIR / f'processed_final_data_seed_{SEED}_train_{TRAIN_RATIO}_test_{TEST_RATIO}.pt'

    if not split_path.is_file():
        splitter = SplitSampler(train_split=TRAIN_RATIO, test_split=TEST_RATIO, validation_split=VALIDATION_RATIO, seed=SEED)
        splitter.build_split(item_ids=dataset.item_ids, labels=dataset.labels)
        splitter.save_split(split_path)
    else:
        splitter = SplitSampler.load_file(split_path)

    std_scaler_path = DATA_PROCESSED_DIR / f'processed_final_data_seed_{SEED}_train_{TRAIN_RATIO}_standard_sclaer.pt'

    train_ids = splitter.get_split('train')
    test_ids = splitter.get_split('test')
    validation_ids = splitter.get_split('validation')
    if not std_scaler_path.is_file():
        scaler = StandardScaler()
        train_positions = dataset.ids_to_positions(train_ids)
        scaler = scaler.fit(dataset.features[train_positions])
        dataset.features = scaler.transform(dataset.features)

        scaler.save(std_scaler_path)
    else:
        scaler = StandardScaler.load(std_scaler_path)
        dataset.features = scaler.transform(dataset.features)

    train_positions = dataset.ids_to_positions(train_ids)
    train_data = Subset(dataset, train_positions)

    test_data = Subset(dataset=dataset, indices=dataset.ids_to_positions(test_ids))

    validation_data = Subset(dataset=dataset, indices=dataset.ids_to_positions(validation_ids))

    ratio_sampler = RatioSampler(dataset.labels[train_positions], attack_label=1, attack_ratio=ATTACK_RATIO, seed=SEED)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=ratio_sampler,num_workers=0)

    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)
    args = Args()
    model = BaseNormalAutoEncoder(len(dataset[0][0]),args).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    datetime_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    time_name = time.time()
    best_model_settings, (training_log, training_benign_log, training_attack_log, training_contrastive_log), \
            (validation_log, validation_benign_log, validation_attack_log, validation_contrastive_log), \
            (test_log, test_benign_log, test_attack_log, test_contrastive_log) = base_training_contrastive(training_loader=train_loader,
                                                                                val_loader=validation_loader,
                                                                                test_loader=test_loader,
                                                                                model=model,
                                                                                optimizer=optimizer,
                                                                                epoch=EPOCH,device=DEVICE)
    end_time = time.time() - time_name

    print(f"Training Took {end_time} seconds")
    training_log_dir = TRAINING_LOGS_DIR / f'{model.__class__.__name__} {datetime_string}'
    training_log_dir.mkdir(exist_ok=True)
    pl.DataFrame({'train_log': training_log,
                  'train_benign_log': training_benign_log,
                  'train_attack_log': training_attack_log,
                  'train_contrastive_log': training_contrastive_log,
                  'validation_log': validation_log,
                  'validation_benign_log': validation_benign_log,
                  'validation_attack_log': validation_attack_log,
                  'validation_contrastive_log': validation_contrastive_log
                  }).write_csv(training_log_dir / f'{time_name}.csv')
    best_model = np.argmin(validation_log)

    plt.figure(figsize=(10, 10))
    X_range = range(0, len(training_log))
    plt.plot(X_range, training_log, label='Training Loss')
    plt.plot(X_range, validation_log, label='Validation Loss')
    plt.plot(X_range, training_attack_log, label='Training Attack Loss')
    plt.plot(X_range, training_benign_log, label='Training Benign Loss')
    plt.plot(X_range, validation_attack_log, label='Validation Attack Loss')
    plt.plot(X_range, validation_benign_log, label='Validation Benign Loss')
    plt.plot(X_range, training_contrastive_log, label='Training Contrastive Loss')
    plt.plot(X_range, validation_contrastive_log, label='Validation Contrastive Loss')
    plt.axvline(best_model, label='Best Model')
    plt.legend()
    plt.savefig(training_log_dir / f'{time_name}_LOSS_PLOT.png')
    plt.clf()

    with open(training_log_dir / f'{time_name}_details.txt', 'w') as f:
        f.write(f"""
MODEL NAME: {model.__class__.__name__} (attack)
Test Loss: {test_log}
Test Attack Loss: {test_attack_log} (+should be higher)
Test Contrastive Loss: {test_contrastive_log}
Test Benign Loss: {test_benign_log}
Training Time Started: {datetime_string}
Time Taken: {end_time} seconds


# Parameters
EPOCHS: {EPOCH}
EMBEDDING_MODEL: {EMBEDDING_MODEL}
SEED: {SEED}
TRAIN_RATIO: {TRAIN_RATIO}
VALIDATION_RATIO: {VALIDATION_RATIO}
TEST_RATIO: {TEST_RATIO}
BATCH_SIZE: {BATCH_SIZE}
ATTACK_RATIO: {ATTACK_RATIO}

DATASET: {processed_data_only_benign}
        """)


    torch.save(best_model_settings, MODEL_DIRECTORY_DEV / f'{model.__class__.__name__}_attack_{time_name}.pt')
