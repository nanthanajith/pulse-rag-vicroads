# 1. Parse dataset -> create qrels & collection
python data.py

# 2. Encode collection
./encode.sh

# 3. Index collection
./index.sh

# 4. Retrieve & generate runs
python search.py

# 5. Evaluate RAG
#python eval.py known ../data/qrels.txt ../target/runs/vicroads-dense.txt ../target/runs/vicroads-bm25.txt
python eval.py inferred ../data/qrels.txt ../target/runs/vicroads-dense.txt ../target/runs/vicroads-bm25.txt

# 6. Start chatbot
python chatbot.py