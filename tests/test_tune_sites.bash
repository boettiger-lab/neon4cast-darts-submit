#!/bin/bash
> logs/test_tune_sites.log
sites=("ARIK" "BARC" "BIGC" "BLDE" "BLUE" "BLWA" "CARI" 
       "COMO" "CRAM" "CUPE" "FLNT" "GUIL" "HOPB" "KING" 
       "LECO" "LEWI" "LIRO" "MART" "MAYF" "MCDI" "MCRA" 
       "OKSR" "POSE" "PRIN" "PRLA" "PRPO" "REDB" "SUGG" 
       "SYCA" "TECR" "TOMB" "TOOK" "WALK" "WLOU")

# Iterating over the models listed above
for site in "${sites[@]}"; do
  python -u train_tune.py --model BlockRNN --target temperature --site "$site" \
    --date 2023-03-09 --epochs 1 --tune --test &>> logs/test_tune_sites.log
done