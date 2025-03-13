# get all the directories that start with rel_embeddings_

path_to_rel_embeddings=$1

for dir in $path_to_rel_embeddings*; do
    echo "Running probe for $dir/"
    mkdir -p $dir/temp/outputs
    python run_probes.py $dir/ all > $dir/temp/outputs/out.txt 2> $dir/temp/outputs/err.txt &
done