#!/bin/bash
cd /home/bkai/.claude/jobs/1c9cf7e9/tmp
/home/bkai/miniconda3/envs/fastgs2/bin/python AUDIT_cpu_score.py \
 /mnt/d/avv/ADVC/K1_blur05=ADVC_K1blur05 \
 /mnt/d/avv/ADVC/mean6_enc=ADVC_mean6enc \
 /mnt/d/avv/ADVC/K1_enc=ADVC_K1enc \
 /mnt/d/avv/ADVC/meanpair_seed=ADVC_pairseed \
 /mnt/d/avv/ADVC/mean6=ADVC_mean6
