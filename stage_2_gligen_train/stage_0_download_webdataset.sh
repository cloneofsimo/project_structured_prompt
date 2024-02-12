img2dataset --url_list /root/bigdisk/project_structured_prompt/stage_2_gligen_train/GRIT/grit-20m --input_format "parquet" \
    --url_col "url" --caption_col "caption" --output_format files \
    --output_folder ./grit_files --processes_count 4 --thread_count 64 --image_size 768 \
    --resize_only_if_bigger=False --resize_mode="keep_ratio" --skip_reencode=True \
    --save_additional_columns '["id","noun_chunks","ref_exps","clip_similarity_vitb32","clip_similarity_vitl14"]' \
    --enable_wandb False
