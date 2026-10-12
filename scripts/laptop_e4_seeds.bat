@echo off
REM E4 seed repeat (laptop): A x3 -> R x3 -> B x3, 125 evals each
REM   A = paper Figure 1 procedure (--paper-mode)
REM   R = random-only control: LHS 125 within paper theta_base +-10% (broad)
REM   B = A + move to in-box surface max when stationary point is a saddle (--move boxmax, our variant)
REM Start: paper Table 8 theta_D004,base. Tiles: D004 paper 50.
REM Run from C:\rsm_calib :  scripts\laptop_e4_seeds.bat
set PYTHONUTF8=1
set F=D:\flair1_hf\extracted\train\D004_2021
set COM=--img-dir %F%\aerial --msk-dir %F%\labels --tiles docs\paper_tiles\e4_d004_tiles50.txt --checkpoint models\rgb15\FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth --n-classes 19 --label-map labelmap_19.json --targets 5 --base-mode given --theta0 87.32 96.64 90.50 46.47 38.61 33.39
set SRCH=--paper-mode --calib-tiles all --adj0 0.1 --eps-stop -1000 --max-evals 125

call :A 0
call :A 1
call :A 2
call :R 0
call :R 1
call :R 2
call :B 0
call :B 1
call :B 2

git add -f out_e4_A_seed0_n125 out_e4_A_seed1_n125 out_e4_A_seed2_n125 out_e4_R_seed0_n125 out_e4_R_seed1_n125 out_e4_R_seed2_n125 out_e4_B_seed0_n125 out_e4_B_seed1_n125 out_e4_B_seed2_n125
git commit -m "E4 seed repeat A/R/B 125 evals (laptop)"
git pull --rebase
git push
echo DONE
pause
exit /b 0

:A
echo === E4-A seed %1
python run.py calibrate %COM% %SRCH% --seed %1 --out out_e4_A_seed%1_n125
exit /b 0

:R
echo === E4-R seed %1
python run.py broad %COM% --calib-tiles all --adjs 0.1 --n-broad 125 --seed %1 --out out_e4_R_seed%1_n125
exit /b 0

:B
echo === E4-B seed %1
python run.py calibrate %COM% %SRCH% --move boxmax --seed %1 --out out_e4_B_seed%1_n125
exit /b 0
