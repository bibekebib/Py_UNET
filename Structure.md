📦Py_UNET
 ┣ 📂.git
 ┃ ┣ 📂hooks
 ┃ ┃ ┣ 📜applypatch-msg.sample
 ┃ ┃ ┣ 📜commit-msg.sample
 ┃ ┃ ┣ 📜fsmonitor-watchman.sample
 ┃ ┃ ┣ 📜post-update.sample
 ┃ ┃ ┣ 📜pre-applypatch.sample
 ┃ ┃ ┣ 📜pre-commit.sample
 ┃ ┃ ┣ 📜pre-merge-commit.sample
 ┃ ┃ ┣ 📜pre-push.sample
 ┃ ┃ ┣ 📜pre-rebase.sample
 ┃ ┃ ┣ 📜pre-receive.sample
 ┃ ┃ ┣ 📜prepare-commit-msg.sample
 ┃ ┃ ┣ 📜push-to-checkout.sample
 ┃ ┃ ┗ 📜update.sample
 ┃ ┣ 📂info
 ┃ ┃ ┗ 📜exclude
 ┃ ┣ 📂logs
 ┃ ┃ ┣ 📂refs
 ┃ ┃ ┃ ┣ 📂heads
 ┃ ┃ ┃ ┃ ┗ 📜main
 ┃ ┃ ┃ ┗ 📂remotes
 ┃ ┃ ┃ ┃ ┗ 📂origin
 ┃ ┃ ┃ ┃ ┃ ┗ 📜HEAD
 ┃ ┃ ┗ 📜HEAD
 ┃ ┣ 📂objects
 ┃ ┃ ┣ 📂info
 ┃ ┃ ┗ 📂pack
 ┃ ┃ ┃ ┣ 📜pack-4b24718832b0e4fcd78944f26551557f2ab7d220.idx
 ┃ ┃ ┃ ┗ 📜pack-4b24718832b0e4fcd78944f26551557f2ab7d220.pack
 ┃ ┣ 📂refs
 ┃ ┃ ┣ 📂heads
 ┃ ┃ ┃ ┗ 📜main
 ┃ ┃ ┣ 📂remotes
 ┃ ┃ ┃ ┗ 📂origin
 ┃ ┃ ┃ ┃ ┗ 📜HEAD
 ┃ ┃ ┗ 📂tags
 ┃ ┣ 📜HEAD
 ┃ ┣ 📜config
 ┃ ┣ 📜description
 ┃ ┣ 📜index
 ┃ ┗ 📜packed-refs
 ┣ 📂Models
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜unet_layer.cpython-311.pyc
 ┃ ┃ ┗ 📜unet_model.cpython-311.pyc
 ┃ ┣ 📜unet_layer.py
 ┃ ┗ 📜unet_model.py
 ┣ 📂__pycache__
 ┃ ┣ 📜app.cpython-311.pyc
 ┃ ┣ 📜evaluate.cpython-311.pyc
 ┃ ┣ 📜predict.cpython-311.pyc
 ┃ ┗ 📜train.cpython-311.pyc
 ┣ 📂archive
 ┃ ┣ 📂.ipynb_checkpoints
 ┃ ┃ ┗ 📜meta_data-checkpoint.csv
 ┃ ┣ 📂Forest Segmented
 ┃ ┃ ┗ 📂Forest Segmented
 ┃ ┃ ┃ ┣ 📂images
 ┃ ┃ ┃ ┃ ┣ 📜10452_sat_08.jpg
 ┃ ┃ ┃ ┃ ┣ 📜10452_sat_18.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_00.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_01.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_02.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_03.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_04.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_07.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_08.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_10.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_12.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_13.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_sat_15.jpg
 ┃ ┃ ┃ ┣ 📂masks
 ┃ ┃ ┃ ┃ ┣ 📜10452_mask_08.jpg
 ┃ ┃ ┃ ┃ ┣ 📜10452_mask_18.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_00.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_01.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_02.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_03.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_04.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_07.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_08.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_10.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_12.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_13.jpg
 ┃ ┃ ┃ ┃ ┣ 📜111335_mask_15.jpg
 ┃ ┃ ┃ ┗ 📜meta_data.csv
 ┃ ┗ 📜meta_data.csv
 ┣ 📂outputs
 ┃ ┣ 📂.ipynb_checkpoints
 ┃ ┃ ┗ 📜df_folder-checkpoint.csv
 ┃ ┣ 📂graphs
 ┃ ┃ ┣ 📜train_val_accuracy.jpg
 ┃ ┃ ┗ 📜train_val_loss.jpg
 ┃ ┣ 📂model
 ┃ ┃ ┗ 📜model.pth
 ┃ ┣ 📂report
 ┃ ┃ ┣ 📜outout_values.json
 ┃ ┃ ┣ 📜outout_values.txt
 ┃ ┃ ┣ 📜output_values.csv
 ┃ ┃ ┣ 📜output_values.json
 ┃ ┃ ┣ 📜project_report.md
 ┃ ┃ ┗ 📜summary_log.md
 ┃ ┣ 📂result_image
 ┃ ┃ ┣ 📜test_image.jpg
 ┃ ┃ ┗ 📜testimage.jpg
 ┃ ┣ 📜.DS_Store
 ┃ ┣ 📜df_folder.csv
 ┃ ┣ 📜df_test_folder.csv
 ┃ ┣ 📜df_train_folder.csv
 ┃ ┗ 📜test.csv
 ┣ 📂test_images
 ┃ ┗ 📜pexels-johannes-plenio-1423600.jpg
 ┣ 📂utils
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜dice_loss.cpython-311.pyc
 ┃ ┃ ┣ 📜load_data.cpython-311.pyc
 ┃ ┃ ┣ 📜misc.cpython-311.pyc
 ┃ ┃ ┣ 📜report_ai.cpython-311.pyc
 ┃ ┃ ┗ 📜transforms.cpython-311.pyc
 ┃ ┣ 📜dice_loss.py
 ┃ ┣ 📜generate_log.py
 ┃ ┣ 📜load_data.py
 ┃ ┣ 📜misc.py
 ┃ ┣ 📜report_ai.py
 ┃ ┗ 📜transforms.py
 ┣ 📜.DS_Store
 ┣ 📜.gitignore
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜app.py
 ┣ 📜config.json
 ┣ 📜data.csv
 ┣ 📜evaluate.py
 ┣ 📜predict.py
 ┣ 📜run.py
 ┗ 📜train.py