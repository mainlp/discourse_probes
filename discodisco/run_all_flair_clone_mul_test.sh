for CORPUS_NAME in `ls data/2023 | sort | tac`; do
  echo "#@#@ $CORPUS_NAME"
  bash rel_scripts/run_single_flair_clone_test.sh $CORPUS_NAME ${1:-models}
done
