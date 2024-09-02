# H3MAE
Pytorch implementation of paper:  Hierarchical Multi-Scale Matched Masked Autoencoder for Multi-Rate Time Series Modeling  
(Submitted for blind review.)

## Data
The raw datasets should be put into `dataset_generation/` folder.

## Usage
### H3MAE

To pre-train H3MAE on a dataset, run the following command:
```Pre-train H3MAE
python main.py  --fine_tune False  --config_path <config_path> --selected_dataset <dataset_name> --multi_rate_groups <groups> 
```
To fine-tune H3MAE on a dataset, run the following command:
```Fine-tune H3MAE
python main.py  --fine_tune True  --pretrain_dic <pretrain_dic> --config_path <config_path> --selected_dataset <dataset_name> --multi_rate_groups <groups>
```
### Compared temporal neural networks 
Run updown_train.py

### Compared self-supervised methods 
Run updown_train_self.py

### MSRL-TA
Run MSRLTA_train.py
