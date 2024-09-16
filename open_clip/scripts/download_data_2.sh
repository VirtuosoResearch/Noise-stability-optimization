img2dataset --url_list cc3m_validation.tsv --input_format "tsv"\
    --url_col "url" --caption_col "caption" --output_format webdataset\
    --output_folder cc3m_validation --processes_count 16 --thread_count 64 --image_size 256