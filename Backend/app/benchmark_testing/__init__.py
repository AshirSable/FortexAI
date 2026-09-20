from pathlib import Path

from ml_factory import DATA_RAW_DIR
from ml_factory.datasets.sampler import SplitSampler
from ml_factory.utils import give_id_to_data
import polars as pl

BENCHMARK_RESULTS = Path(__file__).parent / "bench_mark_results"

from app.benchmark_testing.bench_mark1 import benchmark_pipeline_on_ood
from app.benchmark_testing.bench_mark2 import run_poisoning_experiment

BENCHMARK_RESULTS.mkdir(exist_ok=True)

if __name__ == "__main__":
    data = pl.read_parquet(DATA_RAW_DIR / "shieldlm_ood_dataset.parquet")
    data = give_id_to_data(data)
    sampler = SplitSampler(
        train_split=0.0, test_split=0.8, validation_split=0.2, seed=3123
    )
    sampler.build_split(data["id"], data["label"])

    test_data = data.filter(pl.col("id").is_in(sampler.get_split("test")))
    validation_data = data.filter(pl.col("id").is_in(sampler.get_split("validation")))

    test_data_limit = test_data.limit(500)

    print("Created Test Data of OOD Dataset")
    print("bench marking with 50 samples")

    # benchmark_pipeline_on_ood(
    #     test_data["text"],
    #     test_data["label"],
    #     test_data["label_category"],
    #     out_path=BENCHMARK_RESULTS / "bench_mark.csv",
    # )
    #
    # print("bench mark 1 completed")

    run_poisoning_experiment(
        test_data_limit["text"].to_list(),
        test_data_limit["label"].to_list(),
        poison_prompts=validation_data.filter(pl.col("label") == 0)
        .limit(100)["text"]
        .to_list(),
        out_path=BENCHMARK_RESULTS / "benchmark2_poisoning_results.csv",
    )
    print("Bench mark 2 completed")
