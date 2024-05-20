# Bart pretrain
cd ../generator/
CUDA_VISIBLE_DEVICES=0 python3 pretrain_sci.py\
    --generator_model_name t5-small\
    --train_table_text_path ../retriever/tsdae_data/train.source\
    --train_reference_sentence_path ../retriever/tsdae_data/train.target\
    --dev_table_text_path ../retriever/tsdae_data/dev.source\
    --dev_reference_sentence_path ../retriever/tsdae_data/dev.target\
    --dev_reference_path ../retriever/tsdae_data/dev.target\
    --special_token_path ../../sciGen_data/sciGen_vocab.txt\
    --ckpt_path ../Baselines/outputs/ckpt/\
    --inference_output_file t5_pretrain_output.txt\
    --max_table_len 512\
    --max_tgt_len 512\
    --max_decode_len 512\
    --batch_size 8\
    --total_steps 300000\
