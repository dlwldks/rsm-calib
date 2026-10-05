@echo off
REM 10/5 노트북: 순서 1~3 (E5 1·2단계, E4·E5 비율 전이, E4·E5 MiT-B5) 을 한 번에 실행
REM 모두 논문 Table 8·9 의 θ를 그대로 넣고 추론만 한다 (탐색 없음).
REM 실행: C:\rsm_calib 에서  scripts\laptop_e5_transfer_mitb5.bat
REM 결과: out_e5_* / out_e4_* 폴더 (각각 apply.json, iou_table.csv)

set F=D:\flair1_hf\extracted\train
set R34=models\rgb15\FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth
set MB5=models\mitb5\FLAIR-INC_rgb_15cl_mitb5-unet_weights.pth
set D4=--img-dir %F%\D004_2021\aerial --msk-dir %F%\D004_2021\labels --tiles docs\paper_tiles\e4_d004_tiles50.txt
set D67=--img-dir %F%\D067_2021\aerial --msk-dir %F%\D067_2021\labels --tiles docs\paper_tiles\e5_d067_tiles150.txt
set COM=--n-classes 19 --label-map labelmap_19.json --targets 5

echo === [1] E5 1단계: 논문 θbase (기대 침엽수 75.93)
python run.py apply %D67% --checkpoint %R34% %COM% --theta 97.28 108.99 109.48 62.47 56.64 54.35 --out out_e5_paper_base

echo === [1] E5 2단계: 논문 θbest (기대 80.35)
python run.py apply %D67% --checkpoint %R34% %COM% --theta 98.05 114.46 103.70 62.32 56.19 53.94 --out out_e5_paper_best

echo === [2] E4 비율 전이: D067 비율로 만든 D004 θ (기대 42.61)
python run.py apply %D4% --checkpoint %R34% %COM% --theta 88.01 101.49 85.71 46.36 38.30 33.13 --out out_e4_transfer

echo === [2] E5 비율 전이: D004 비율로 만든 D067 θ (기대 80.41, 본문 115.87)
python run.py apply %D67% --checkpoint %R34% %COM% --theta 100.69 115.87 104.32 63.06 55.77 54.11 --out out_e5_transfer

echo === [2] E5 비율 전이: Table 9 표기값 G=116.87 로도 확인
python run.py apply %D67% --checkpoint %R34% %COM% --theta 100.69 116.87 104.32 63.06 55.77 54.11 --out out_e5_transfer_g116

echo === [3] E4 MiT-B5: ResNet34 최적 θ 적용 (기대 33.48)
python run.py apply %D4% --checkpoint %MB5% --encoder mit_b5 %COM% --theta 90.39 103.62 86.23 46.91 38.01 33.23 --out out_e4_mitb5

echo === [3] E5 MiT-B5: ResNet34 최적 θ 적용 (기대 62.19)
python run.py apply %D67% --checkpoint %MB5% --encoder mit_b5 %COM% --theta 98.05 114.46 103.70 62.32 56.19 53.94 --out out_e5_mitb5

echo === 끝. 결과 올리기:
echo git add -f out_e5_paper_base out_e5_paper_best out_e4_transfer out_e5_transfer out_e5_transfer_g116 out_e4_mitb5 out_e5_mitb5
echo git commit -m "E5 1·2단계, 비율 전이, MiT-B5 결과 (노트북)" ^&^& git pull --rebase ^&^& git push
