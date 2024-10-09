This is the code for the paper "Decision Focused Causal Learning for Direct Counterfactual Marketing Optimization".

1. Pytorch 2.0.1 is used in this experiment.

2. The directory structure is shown as follows.

|-----DFCL_code
|     |-----criteo_uplift_experiment.ipynb       # A demo to run DFCL (including IFD, PPL, MER) in CRITEO-UPLIFT v2
|     |-----Lagrangian_duality_gradient_estimator.py       # The gradient estimator of IFD
|     |-----Metric.py     		 # The evaluation metrics
|     |-----model.py                 # The uplift model for CRITEO-UPLIFT v2
|     |-----utils.py                   # Processing data
|     |-----README.txt               
|     |-----Model                               
|     	    |-----model_before_finetune.pkl                            # A trained two-stage model weights

3. In the experiments of CRITEO-UPLIFT v2, we fine-tune the trained two-stage model using only decision loss (in the demo we provide a trained two-stage model with AUCC = 0.7576). In the marketing data, we use decision loss and predictive loss for joint training.

5. Download the dataset named CRITEO-UPLIFT v2 from https://ailab.criteo.com/criteo-uplift-prediction-dataset/. You can put this dataset in the data directory, and rename it as "criteo-uplift-v2.1.csv". By this way, you can run the demo in the code directory based on this dataset.

6. Due to data privacy, the real business marketing data used in this experiment is not provided. 