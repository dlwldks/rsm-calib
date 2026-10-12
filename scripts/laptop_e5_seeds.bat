@echo off
REM E5 seed repeat (laptop): A = paper Figure 1 procedure, R = random-only control. 125 evals each.
REM Same settings as analysis/results/e5/search_A_seed0_n125 (seed 0 already done).
REM Start: paper Table 9 theta_D067,base. Tiles: D067 paper 150.
REM Run from C:\rsm_calib :  scripts\laptop_e5_seeds.bat
set PYTHONUTF8=1
set F=D:\flair1_hf\extracted\train\D067_2021
set COM=--img-dir %F%\aerial --msk-dir %F%\labels --tiles docs\paper_tiles\e5_d067_tiles150.txt --checkpoint models\rgb15\FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth --n-classes 19 --label-map labelmap_19.json --targets 5 --base-mode given --theta0 97.28 108.99 109.48 62.47 56.64 54.35
set SRCH=--paper-mode --calib-tiles all --adj0 0.1 --eps-stop -1000 --max-evals 125

call :A 1
call :A 2
call :R 0
call :R 1
call :R 2

git add -f out_e5_A_seed1_n125 out_e5_A_seed2_n125 out_e5_R_seed0_n125 out_e5_R_seed1_n125 out_e5_R_seed2_n125
git commit -m "E5 seed repeat A/R 125 evals (laptop)"
git pull --rebase
git push
echo DONE
pause
exit /b 0

:A
echo === E5-A seed %1
python run.py calibrate %COM% %SRCH% --seed %1 --out out_e5_A_seed%1_n125
exit /b 0

:R
echo === E5-R seed %1
python run.py broad %COM% --calib-tiles all --adjs 0.1 --n-broad 125 --seed %1 --out out_e5_R_seed%1_n125
exit /b 0
