bash train_quest.sh 2 none llama adamw; # row 1
bash train_quest_tgcs_ablation.sh 2 none llama adamw 7 0; # row A
bash train_quest_tgcs_ablation.sh 2 none llama adamw 0 0; # row B
bash train_quest_ablation.sh 2 none llama adamw 0 0; # row C
bash train_quest_ablation.sh 2 gaussian llama adamw 0 0; # row 2
bash train_quest_tgcs.sh 2 gaussian llama adamw; # row 3
bash train_quest_tgcs_ablation.sh 2 gaussian llama adamw 7 0; # row D
bash train_quest_ablation.sh 2 gaussian llama adamw 7 0; # row E
bash train_quest_tgcs_ablation.sh 2 gaussian llama adamw 7 7; # row 4
bash train_bbq.sh 2 none llama adamw; # row 5
bash train_bbq_tgcs_ablation.sh 2 none llama adamw 7 0; # row F
bash train_bbq_tgcs_ablation.sh 2 none llama adamw 0 0; # row G
bash train_bbq_ablation.sh 2 none llama adamw 0 0; # row H
bash train_bbq_ablation.sh 2 gaussian llama adamw 0 0; # row I
bash train_bbq_tgcs.sh 2 gaussian llama adamw; # row 7
bash train_bbq_tgcs_ablation.sh 2 gaussian llama adamw 7 0; # row J
bash train_bbq_ablation.sh 2 gaussian llama adamw 7 0; # row K
bash train_bbq_tgcs_ablation.sh 2 gaussian llama adamw 7 7; # row 8