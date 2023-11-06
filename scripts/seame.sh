
dataset="seame"
model="large"
dataset_dir="/kaggle/input/seame-conversation-phase1/phase1"
single_lang_threshold=1
concat_lang_token=1
code_switching="zh-en"
# need both concat_lang_token to be 1 and code_switching to be "zh-en" to enable lang concat in the prompt
# only turn code-switching to be "zh-en" will do normal whisper LID to select language token for the prompt
# if code-switching is "0", you should pass in a language token e.g. "zh", and we will therefore use this for all utterances

echo "currently testing ${model}"
python ../csasr_st.py \
--single_lang_threshold ${single_lang_threshold} \
--concat_lang_token ${concat_lang_token} \
--code_switching ${code_switching} \
--model ${model} \
--dataset ${dataset} \
--dataset_dir ${dataset_dir} \
--beam_size 5 \
--topk 1000 \
--task transcribe 
