# Quantization-Aware Training with Constrained Empirical Weight Distribution

This is the implementation of our work "Quantization-Aware Training with Constrained Empirical Weight Distribution".
We provide scripts that allows readers to recreate all data in our figures and tables.

### Setup
```bash
pip install -r requirements.txt
```

### Code Structure and Sources
Our code under is adapted from [QuEST](https://github.com/IST-DASLab/QuEST/tree/main) with a few modifications. The vision code in `vision/` is adapted from [Facebook/DeiT](https://github.com/facebookresearch/deit). 

The most important pieces of code are as follows:
1. The code for CEWT is in `./optimizer/project.py`
1. The code for optimizers with CEWT integration is under `./optimizer/*.py`
2. The code for quantizers with CEWT integration is in `./models/quantization/base_linear.py`

### Figure 1: Effects of CEWT
First, run the following experiments.
```bash
bash train_bbq.sh 2 none llama adamw;
bash train_bbq_tgcs.sh 2 gaussian llama adamw;
bash train_quest.sh 2 none llama adamw;
bash train_quest_tgcs.sh 2 gaussian llama adamw;
bash train_lsq.sh 2 none llama adamw;
bash train_lsq.sh 2 gaussian llama adamw;
```
Then, find your checkpoints under `./exps` and edit `src/plots.ipynb` as necessary.


### Figure 2: Visualization of CEWT
The script to generate this figure can be found in `src/toys2.ipynb`.

### Table 2: Comparison between weight oscillation Suppression Methods
The script to generate the data in this table can be found in `src/toys2.ipynb`.

### Table 3: Evaluation Perplexity of LLaMA and GPT models pretrained on C4.
Run the following experiments to recreate the 95M rows.
```bash
for model in base llama; do
      for bits in 2 1; do
            for opt in muon adamw lion sso; do
                  bash train_bbq.sh $bits none $model $opt;
                  bash train_bbq_tgcs.sh $bits gaussian $model $opt;
                  bash train_quest.sh $bits none $model $opt;
                  bash train_quest_tgcs.sh $bits gaussian $model $opt;
                  bash train_lsq.sh $bits none $model $opt;
                  bash train_lsq.sh $bits gaussian $model $opt;
            done
      done
done
```
To reproduce results for larger experiments, edit the scripts to comment the following lines.
Note that 30M here refers to non-embedding parameters.
```bash
# 30M
export N_LAYER=6
export N_EMBD=640
export N_HEAD=5
export LR=0.0012
export TOKENS=3000000000 # 3B
export MODEL_SIZE_PREFIX="30M"
export MATRIX_LR=0.01
```
To recreate 125M results, uncomment the following. Note that 50M here refers to non-embedding parameters.
```bash
# # 50M
# export N_LAYER=7
# export N_EMBD=768
# export N_HEAD=6
# export LR=0.0012
# export TOKENS=5000000000 # 5B
# export MODEL_SIZE_PREFIX="50M"
# export MATRIX_LR=0.01
```
To recreate 200M results, uncomment the following. Note that 100M here refers to non-embedding parameters.
```bash
# # 100M
# export N_LAYER=8
# export N_EMBD=1024
# export N_HEAD=8
# export LR=0.0006
# export TOKENS=10000000000 # 10B
# export MODEL_SIZE_PREFIX="100M"
# export MATRIX_LR=0.005
```
To recreate 300M results, uncomment the following. Note that 200M here refers to non-embedding parameters.
```bash
# # 200M
# export N_LAYER=10
# export N_EMBD=1280
# export N_HEAD=10
# export LR=0.0003
# export TOKENS=20000000000 # 20B
# export MODEL_SIZE_PREFIX="200M"
# export MATRIX_LR=0.0025
```
To recreate 600M results, uncomment the following. Note that 430M here refers to non-embedding parameters.
```bash
# # 430M
# export N_LAYER=13
# export N_EMBD=1664
# export N_HEAD=13
# export LR=0.00015
# export TOKENS=43000000000 # 43B
# export MODEL_SIZE_PREFIX="430M"
# export MATRIX_LR=0.00125
```

### Figures 3 and 4: Examples of Rounding Boundary Weight Oscillation
The script to generate these two figures can be found in `src/toys2.ipynb`

### Table 4: Robustness of CEWT over different values of R
Run the following experiments:
```bash
for init in lecun torch spectral_mu_p none; do
    bash train_bbq_sweep_init_schemes.sh 2 none llama adamw $init;
    bash train_bbq_tgcs_sweep_init_schemes.sh 2 gaussian llama adamw $init;
done
```

### Figure 5: RBM, EMAOF, LR, and Perplexity During Training
Run the following experiments and check wandb for the metrics.
```bash
bash train_bbq_log_ema_os_freq.sh 2 none llama adamw 0.01;
bash train_bbq_tgcs_log_ema_os_freq.sh 2 gaussian llama adamw 0.01;
bash train_quest_log_ema_os_freq.sh 2 none llama adamw 0.01;
bash train_quest_tgcs_log_ema_os_freq.sh 2 gaussian llama adamw 0.01;
```

### Table 6: CEWT on Different Rounding Jacobian Estimators
The script to generate the data in this table can be found in `src/toys2.ipynb`

### Tables 7, 8, and 9: When is CEWT less Effective?
Refer to `src/toys2.ipynb`.

### Table 10 and Figure 6: Ablation Studies on Removing HT and TGCS
First, run the following experiments.
```bash
bash train_quest.sh 2 none llama adamw; # row 1
bash train_quest_ablation.sh 2 gaussian llama adamw 0 0; # row 2
bash train_quest_tgcs.sh 2 gaussian llama adamw; # row 3
bash train_quest_tgcs_ablation.sh 2 gaussian llama adamw 7 7; # row 4
bash train_bbq.sh 2 none llama adamw; # row 5
bash train_bbq_ablation.sh 2 gaussian llama adamw 0 0; # row 6
bash train_bbq_tgcs.sh 2 gaussian llama adamw; # row 7
bash train_bbq_tgcs_ablation.sh 2 gaussian llama adamw 7 7; # row 8
```
To generate the figure, refer to `src/plots2.ipynb` once the experiments are done.

### Figure 7: Periodic Interpolation
Run these experiments to recreate the data.
```bash
bash train_quest.sh 2 none llama adamw;
bash train_quest_tgcs.sh 2 gaussian llama adamw; 
for pi_alpha in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8; do
     bash train_quest_pi.sh 2 none llama adamw $pi_alpha; 
done
for dl_alpha in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8; do
     bash train_quest_dl.sh 2 none llama adamw $dl_alpha; 
done
```

### Figure 8: Dampening loss
Run these experiments to recreate the data.
```bash
bash train_quest.sh 2 none llama adamw;
bash train_quest_tgcs.sh 2 gaussian llama adamw; 
for dl_alpha in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8; do
     bash train_quest_dl.sh 2 none llama adamw $dl_alpha; 
done
```

### Table 11: PQN
Run these experiments to recreate the data.
```bash
for bits in 1 2 3; do 
      bash train_bbq.sh $bits none llama adamw;
      bash train_bbq_pqn.sh $bits none llama adamw;
      bash train_bbq_tgcs.sh $bits gaussian llama adamw;
      bash train_quest.sh $bits none llama adamw;
      bash train_quest_pqn.sh $bits none llama adamw;
      bash train_quest_tgcs.sh $bits gaussian llama adamw;
done
```

### Table 12: Full-Precision Perplexity
Run these experiments to recreate the data.
```bash
bash train_base.sh;
bash train_base_ho.sh;
```
For larger models, refer to the Section on Table 3.

### Table 13: Zero-shot Evaluation
Refer to the Section on Table 3 for the pre-training runs first. Suppose the produced checkpoint of a particular experiment is at `exps/<exp_name>`,
execute the following command
```bash
python3 src/eval_hswag.py --model_name exps/<exp_name>
```
This will generate json files with zero-shot results under `exps/<exp_name>`. Do this for all experiment checkpoints.

### Table 14: Training Time
Refer to the Section on Table 3 for the pre-training runs first. Then check on W&B for training time.

### Table 15 and 16: Vision Models
Run the following experiments.
```bash
cd vision;
for bits in 1 2; do
      bash train_bbq.sh $bits none NoQuantizer 300;
      bash train_bbq_tgcs.sh $bits gaussian NoQuantizer 300;
      bash train_quest.sh $bits none NoQuantizer 300;
      bash train_quest_tgcs.sh $bits gaussian NoQuantizer 300;
      
      bash train_bbq.sh $bits none BBQ 300;
      bash train_bbq_tgcs.sh $bits gaussian BBQ 300;
      bash train_quest.sh $bits none QuESTImproved 300;
      bash train_quest_tgcs.sh $bits gaussian QuESTImproved 300;

      bash train_bbq.sh $bits none BBQ 600;
      bash train_bbq_tgcs.sh $bits gaussian BBQ 600;
      bash train_quest.sh $bits none QuESTImproved 600;
      bash train_quest_tgcs.sh $bits gaussian QuESTImproved 600;
done
```