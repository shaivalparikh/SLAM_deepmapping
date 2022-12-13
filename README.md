
Our implementation for the "DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds" paper. 

Steps to run the repository:

To extract 2D Data:
```bash
  tar -xvf ./data/2D/all_poses.tar -C ./data/2D/
```

To train in unsupervised manner: 
```bash
  run_train_2D.sh
```

To train using warmstart approach:
```bash
  run_icp.sh
```

To run evaluation script:
```bash
  run_eval_2D.sh
```



