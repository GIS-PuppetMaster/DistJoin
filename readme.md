This is the source code of paper ****DistJoin: A Decoupled Join Cardinality Estimator based on Adaptive Neural Predicate Modulation****

## Env setup
1. Install python3.12
2. Install required packages in requirements.txt `python install -r requirements.txt`
3. Install our sampler package that response for generating training data dynamically during training `python install ./MySampler/setup.py install`

## Prepare Dataset
1. Put the JOB datasets into `./datasets/job`, all csv table should have headers
2. You can update the true cards by first removing all `{wokload}.pkl` file in `./queries` and run the `./queries/ConvertMSCNTestWorkload.py`, which will automatically calculates true cards and convert the test workloads to MSCN's format
3. Use `./queries/GetJoinWithoutPredicatesCard.py` to pre-calculates the cardinality of queries' join schemas if needed

## Setup experiments
1. Use `./Configs/IMDB/IMDB.yaml` to set experiments, or you can use the default one to perform our experiments in the paper

## Train DistJoin
1. Run `python train.py`
2. Copy the `exp mark` in the output for latter testing, which is a timestamp

## Test DistJoin 
1. Run `python eval-IMDB-all.py --config=IMDB --no_wandb` and enter the `exp mark` to evaluate the workloads configurated in the `IMDB.yaml` file, it will cover all five join conditions on that workload
2. Check the results in the output and the ./results/DistJoin