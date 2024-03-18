gdown https://drive.google.com/drive/folders/1RYsW52zfvAkXXioqghLNUs-8lQyz1Gvp?usp=sharing -O ./ --folder
cd data
for zip_file in *.zip; do unzip "${zip_file%%.zip}"; done 