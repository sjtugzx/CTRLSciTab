# bart without background information & highlight cell
cd ../generator/
CUDA_VISIBLE_DEVICES=2 python3 pretrain_sci.py\
    --generator_model_name facebook/bart-base\
    --train_table_text_path ../retriever/tsdae_without_bginfo_highlightcell/train.source\
    --train_reference_sentence_path ../retriever/tsdae_without_bginfo_highlightcell/train.target\
    --dev_table_text_path ../retriever/tsdae_without_bginfo_highlightcell/dev.source\
    --dev_reference_sentence_path ../retriever/tsdae_without_bginfo_highlightcell/dev.target\
    --dev_reference_path ../retriever/tsdae_without_bginfo_highlightcell/dev.target\
    --special_token_path ../../sciGen_data/sciGen_vocab.txt\
    --ckpt_path ../Baselines/outputs/ckpt_without_bginfo_hcell/\
    --inference_output_file bart_wobgwohl_pretrain_out.txt\
    --max_table_len 512\
    --max_tgt_len 512\
    --max_decode_len 512\
    --batch_size 8\
    --total_steps 300000\