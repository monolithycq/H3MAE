# H3MAE
Pytorch implementation of paper:  Hierarchical Multi-Scale Matched Masked Autoencoder for Multi-Rate Time Series Modeling  
(Submitted for blind review.)

## Data
The raw datasets should be put into `dataset_generation/` folder.

## Usage
### H3MAE

```Pre-train H3MAE
python main.py  --fine_tune False  --config_path <config_path> --selected_dataset <dataset_name> --multi_rate_groups <groups> 
```
```Fine-tune H3MAE
python main.py  --fine_tune True  --pretrain_dic <pretrain_dic> --config_path <config_path> --selected_dataset <dataset_name> --multi_rate_groups <groups>
```
