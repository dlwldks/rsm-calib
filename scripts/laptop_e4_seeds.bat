@echo off
REM 10/5 노트북: E4 탐색 seed 반복 (A x3 -> R x3 -> B x3), 모두 추론 125회 고정
REM   A = 논문 Figure 1 절차 그대로 (--paper-mode)
REM   R = 무작위 대조군: 논문 theta_base +-10% 안에서 LHS 125개만 추론 (broad)
REM   B = A + 정상점이 안장점이면 1단계 범위 안 곡면 최대점으로 이동 (--move boxmax, 우리 변형)
REM 출발점: 논문 Table 8 theta_D004,base. 타일: D004 논문 50장.
REM 실행: C:\rsm_calib 에서  scripts\laptop_e4_seeds.bat
REM 끊기면: 해당 줄만 다시 실행하되 calibrate 줄은 끝에 --resume 추가 (같은 --out)
set F=D:\flair1_hf\extracted\train\D004_2021
set COM=--img-dir %F%\aerial --msk-dir %F%\labels --tiles docs\paper_tiles\e4_d004_tiles50.txt --checkpoint models\rgb15\FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth --n-classes 19 --label-map labelmap_19.json --targets 5 --base-mode given --theta0 87.32 96.64 90.50 46.47 38.61 33.39
set SRCH=--paper-mode --calib-tiles all --adj0 0.1 --eps-stop -1000 --max-evals 125

for %%s in (0 1 2) do (
  echo === E4-A seed %%s
  python run.py calibrate %COM% %SRCH% --seed %%s --out out_e4_A_seed%%s_n125
)
for %%s in (0 1 2) do (
  echo === E4-R seed %%s
  python run.py broad %COM% --calib-tiles all --adjs 0.1 --n-broad 125 --seed %%s --out out_e4_R_seed%%s_n125
)
for %%s in (0 1 2) do (
  echo === E4-B seed %%s
  python run.py calibrate %COM% %SRCH% --move boxmax --seed %%s --out out_e4_B_seed%%s_n125
)
echo === 끝. 결과 올리기:
echo git add -f out_e4_A_seed*_n125 out_e4_R_seed*_n125 out_e4_B_seed*_n125
echo git commit -m "E4 seed 반복 A/R/B 125회 (노트북)" ^&^& git pull --rebase ^&^& git push
