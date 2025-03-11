# LaGRAR
Task-aware Latent Graph Rewiring Can Robustly Solve Oversquashing-Oversmoothing Dilemma

To run a node classification experiment on Cora dataset with LaGRAR:

python main.py --task node --dataset cora --model lagrar --encoder gcn --num_layers 2 --hidden 16 --dropout 0.5 --lr 0.01 --epochs 100

To run link prediction across all datasets:

python main.py --task link --model lagrar --run_all_datasets --visualize

To run an ablation study:
python main.py --task node --dataset cora --model lagrar --run_ablation