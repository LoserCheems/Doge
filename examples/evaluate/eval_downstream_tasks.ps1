
$MODEL = "JingzeShi/Doge-20M"
$OUTPUT_DIR = "./lighteval_results"

lighteval accelerate --override_batch_size 1 `
    --model_args "pretrained=$MODEL" `
    --output_dir $OUTPUT_DIR `
    --tasks "original|mmlu|5|1,lighteval|triviaqa|5|1,lighteval|arc:easy|5|1,leaderboard|arc:challenge|5|1,lighteval|piqa|5|1,leaderboard|hellaswag|5|1,lighteval|openbookqa|5|1,leaderboard|winogrande|5|1"