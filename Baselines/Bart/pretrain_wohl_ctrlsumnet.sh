# bart without highlight
cd ../generator/
CUDA_VISIBLE_DEVICES=3 python3 pretrain_sci.py\
    --generator_model_name facebook/bart-base\
    --train_table_text_path ../retriever/tsdae_without_highlightcell/train.source\
    --train_reference_sentence_path ../retriever/tsdae_without_highlightcell/train.target\
    --dev_table_text_path ../retriever/tsdae_without_highlightcell/dev.source\
    --dev_reference_sentence_path ../retriever/tsdae_without_highlightcell/dev.target\
    --dev_reference_path ../retriever/tsdae_without_highlightcell/dev.target\
    --special_token_path ../../sciGen_data/sciGen_vocab.txt\
    --ckpt_path ../Baselines/outputs/ckpt_without_hcell/\
    --inference_output_file bart_wohl_pretrain_out.txt\
    --max_table_len 512\
    --max_tgt_len 512\
    --max_decode_len 512\
    --batch_size 8\
    --total_steps 300000\