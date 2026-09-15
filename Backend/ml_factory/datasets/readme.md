# Datasets

`datasets` module is for managing datasets for multi model data, covering embedding data for autoencoders,
token data for berts and raw text data for LLM's (implemented in future)

This module covers processing, splitting, sampling, and managing / processing of data ready to use for training purposes


## Precomputing

We need to precompute entire given data, before it reaches the model for training
may it be embedding and structural data for autoencoders or tokenization and chunking for BERTS

what stays consistent through out any dataset we would create are columns:

    - `ids` - consisting hashed row id, for data rows, this would help in making deterministic splits across different use of the data
    for example for chunked BERT data and for normal embedding data, we would split on the sampe row id

    - `labels` - consists the labels they have, clearly benign=0 and attack=1


but for other models the dataset covers (atleast for now)

    - AutoEncoder: embedding column (contains contextual embedding), structural column (contains data about some structural patterns found in the data)
    - BERT: text column (contains what text it holds), tokenized (tokenized text),
        if chunking is enabled then chunk_idx contains 0 indexed position for the original text, num_chunks which is a static feature for each chunk/original text that would contain the number of chunks

### Usage Example

**Auto-Encoder**

```python

from ml_factory import DATA_PROCESSED_DIR
from ml_factory.utils import embedding_text
import os
import polars as pl
from ml_factory.utils.structural_extractor import StructuralExtractor
from ml_factory.utils import merge_parts_to_dir, give_id_to_data
from pathlib import Path

# data path to be scanned (LazyFrame)
data = pl.scan_parquet(...)

data = give_id_to_data(data, text_col = 'text', _id_col_name = 'id')

# all of the structural data to extract from text
struc = StructuralExtractor()

outpath = DATA_PROCESSED_DIR / 'out_path.npz'
# process data in parts and returns the parts path
part_paths = precompute_features_ae(
dataset_records = data, # LazyFrame data
embed_fn=embedding_text, # embedding function
embedding_model_name='nomic-embed-text', # embedding model name
out_path= outpath, # outpath
batch_size=32, # batch size
chunk_size=100, # chunks or data points processed togather
max_concurrent=os.cpu_count() or 6, # how many concurrent process do you need ( < cpu count)
structural_extractor = struc, # structural extractor
save_n = 1000, # how many instances to save in file
)

merge_parts(part_paths, out_path) # merge the parts given

```

**BERTs**


```python

from ml_factory import DATA_PROCESSED_DIR
from ml_factory.utils import embedding_text
import os
from pathlib import Path
import polars as pl
from ml_factory.utils.structural_extractor import StructuralExtractor
from ml_factory.utils import merge_parts_to_dir
from ml_factory.datasets.precompute_tokens import precompute_tokens


dataset_path = Path(...)
outpath = DATA_PROCESSED_DIR / 'out_path/'
# process data in parts and returns the parts path
part_paths = precompute_tokens(
dataset_path = dataset_path,
out_path = outpath,
tokenizer = AutoTokenizer,
pretrained_tokenizer='google/bert_uncased_L-2_H-128_A-2',
max_length=128,
save_bytes=100_000_000,
chunk_size=256,
token_chunk=True,
stride=32
)

merge_parts_to_dir(part_paths, out_path) # merge the parts given

```

## Datasets


For now there are 2 different Datasets (as of now)
that are `PromptFeatureDataset` (for autoencoders) and `PromptBERTDataset` (for berts)

`PromptFeatureDataset` takes in .npz file path, outputs 2 tuple values of Features and Labels
`PromptBERTDataset` takes in .npz file or a directory, outputs `PromptBERTResult` which contains tokenized, label, doc_id, text, chunk_idx, num_chunks, attention_masks
this also takes another parameter config_path, which is optional if you have stored the config file in another place

## Sampler and Splitter

The sampler contains a single sampler that is `SplitSampler`, This sampler is deterministic, sampled by hashing method

The sampler can also be saved locally so that you do not have to run it every time.

The samples for train, test, validation split are also transferable across different dataset type (autoencoder or BERT)

and Also samples made from `PromptBERTDataset` or `PromptFeatureDataset` will be the same, considering you used the same seed

```python

# example of PromptBERTDataset

from ml_factory.datasets import PromptBERTDataset
from pathlib import Path
from torch.utils.data import Subset

path_to_dataset = Path(...) # could be .npz or a directory containing .npy files
dataset = PromptBERTDataset(path_to_dataset)

sampler = SplitSampler(train_split=0.7, test_split=0.15, validation_split=0.15)

sampler.build_split(dataset.item_ids, labels=dataset.labels)

train_dataset = Subset(dataset, dataset.ids_to_position(sampler.get_split('train')))

sampler.save_split(Path('split.pt'))

```
To load the split data

```python

sampler.load_file(Path('split.pt'))
```
