cd ../generator/
CUDA_VISIBLE_DEVICES=3 python3 inference_generation.py\
    --generator_model_name facebook/bart-base\
    --train_table_text_path ../retriever/tsdae_data/v2/test.source\
    --train_reference_sentence_path ../retriever/tsdae_data/v2/test.target\
    --dev_table_text_path ../retriever/tsdae_data/v2/test.source\
    --dev_reference_sentence_path ../retriever/tsdae_data/v2/test.target\
    --dev_reference_path ../retriever/tsdae_data/v2/test.target\
    --special_token_path ../../sciGen_data/sciGen_vocab.txt\
    --inference_output_file tsdae_direct_generation_output.txt\
    --max_table_len 512\
    --max_tgt_len 512\
    --max_decode_len 512\
    --batch_size 8\

bert-score -r ../retriever/tsdae_data/v2/test.target -c ../outputs/inference_results/tsdae_direct_generation_output.txt --lang en