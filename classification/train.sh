#!/bin/bash
# nohup $(pwd)/script.sh &
# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"
# python  train_wb.py --cat all --batch 44 --arch eff --mode all --decay 0.01 --vflip False> ~/nohup.out
# python  train_wb.py --cat boobs --batch 32 --arch eff --mode train --decay 0.01 --gray True> ~/nohup.out
# python  train_wb.py --cat poses --batch 44 --arch eff --mode train --decay 0.01 --gray True --vflip False> ~/nohup.out
# python  train_wb.py --cat hair --batch 44 --arch eff --mode train --decay 0.01> ~/nohup.out
# python  train_wb.py --cat body --batch 32 --arch eff --mode train --decay 0.01 --gray True> ~/nohup.out
# python  train_wb.py --cat hair-type --batch 44 --arch eff --mode val --decay 0.01  --gray True> ~/nohup.out
# python  train_wb.py --cat body --batch 32 --arch eff --mode train --decay 0.01> ~/nohup.out
# python  train_wb.py --cat boobs --batch 32 --arch eff --mode train --decay 0.01> ~/nohup.out
# python  train_wb.py --cat tits_size --batch 36 --arch eff --mode all --decay 0.001> ~/nohup.out
# python  train_wb.py --cat hair_type --batch 36 --arch eff --mode all --decay 0.001> ~/nohup.out
# python  train_wb.py --cat sex_positions --batch 36 --arch eff --mode all --decay 0.001> ~/nohup.out
# python  train_wb.py --cat hair_color --batch 36 --arch eff --mode all --decay 0.001> ~/nohup.out
# python  train_wb.py --cat body_decoration_body_painting --batch 32 --arch eff --mode train --decay 0.001> ~/nohup.out
# python  train_wb.py --cat body_decoration_piercing --batch 32 --arch eff --mode train --decay 0.001> ~/nohup.out
# python  train_wb.py --cat body_decoration_tatto --batch 32 --arch eff --mode train --decay 0.001> ~/nohup.out
python  train_wb.py --cat body_type --batch 36 --arch eff --mode all --decay 0.001> ~/nohup.out
