python tools/infer/predict.py --use_gpu=False --use_onnx=True --drop_score=0.7 --det_algorithm="DB" --use_mp=True --total_process_num=2 \\
--image_dir='images/005.jpeg' --det_model_dir=models/model_det.onnx 
--rec_model_dir=models/model_rec.onnx --rec_char_dict_path="models/new_dict.txt" --draw_img_save_dir='results_dir' --vis_font_path='models/latin.ttf'