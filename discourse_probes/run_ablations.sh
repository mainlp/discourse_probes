# get all the directories that start with rel_embeddings_

path_to_rel_embeddings=$1

for dir in $path_to_rel_embeddings*; do
    echo "Running pooling ablation for $dir/"
    mkdir -p $dir/temp/outputs
    python discourse_probes/pooling_ablation.py $dir/ all > $dir/temp/outputs/out.txt 2> $dir/temp/outputs/err.txt &

    if [ $(jobs | wc -l) -ge 8 ]; then
        wait
    fi
done