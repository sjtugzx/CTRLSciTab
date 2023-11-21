# bart without background information
cd ../generator/
CUDA_VISIBLE_DEVICES=1 python3 pretrain_sci.py\
    --generator_model_name facebook/bart-base\
    --train_table_text_path ../retriever/tsdae_without_bginfo/train.source\
    --train_reference_sentence_path ../retriever/tsdae_without_bginfo/train.target\
    --dev_table_text_path ../retriever/tsdae_without_bginfo/dev.source\
    --dev_reference_sentence_path ../retriever/tsdae_without_bginfo/dev.target\
    --dev_reference_path ../retriever/tsdae_without_bginfo/dev.target\
    --special_token_path ../../sciGen_data/sciGen_vocab.txt\
    --ckpt_path ../Baselines/outputs/ckpt_without_bginfo/\
    --inference_output_file bart_wobg_pretrain_out.txt\
    --max_table_len 512\
    --max_tgt_len 512\
    --max_decode_len 512\
    --batch_size 8\
    --total_steps 300000\