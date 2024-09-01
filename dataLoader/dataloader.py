import torch
import torch.nn as nn
import  os
import numpy as np
from utils import DataTransform

class Load_Dataset(nn.Module):
    def __init__(self, dataset, args,augment=False):
        super(Load_Dataset, self).__init__()
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        self.x_data = X_train
        self.y_data = y_train
        self.device = args.device
        self.aug = augment
        if self.aug:
            self.aug1, self.aug2 = DataTransform(self.x_data)


    def __padding__(self):
        origin_len = self.datas[0].shape[0]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        data = torch.tensor(self.x_data[item]).to(self.device)
        label = torch.tensor(self.y_data[item]).to(self.device)
        if self.aug:
            return data, label, torch.tensor(self.aug1[item]).to(self.device),torch.tensor(self.aug2[item]).to(self.device)
        else:
            return data, label

    def shape(self):
        return self.x_data[0].shape


def data_generator(dataset_name, args, augment=False):
    data_path = f"./datasets/{dataset_name}"
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    trainAll_dataset = torch.load(os.path.join(data_path, "train_all.pt"))


    train_dataset = Load_Dataset(train_dataset, args, augment)
    test_dataset = Load_Dataset(test_dataset, args, augment)
    trainAll_dataset = Load_Dataset(trainAll_dataset, args, augment)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,shuffle=True, drop_last=args.drop_last,num_workers=0)
    trainAll_loader = torch.utils.data.DataLoader(dataset=trainAll_dataset, batch_size=args.batch_size,shuffle=True, drop_last=args.drop_last,num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False,num_workers=0)

    return train_loader, test_loader,trainAll_loader

def data_generator_part(dataset_name, part, args, augment=False):
    data_path = f"./datasets/{dataset_name}"
    file_name = "train"+str(part)+".pt"
    train_part_dataset = torch.load(os.path.join(data_path, file_name))
    train_part_dataset = Load_Dataset(train_part_dataset, args, augment)
    train_part_loader = torch.utils.data.DataLoader(dataset=train_part_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, num_workers=0)
    return train_part_loader



def up_sampling(raw_data,max_len,multi_sampling_rates,indexes):
    data = raw_data[:,:,:max_len]  #b,c,l
    group_num = len(multi_sampling_rates)
    input_channels_per_layer = [len(group) for group in multi_sampling_rates]
    start = 0
    list = []
    for i in range(group_num):
        end = start + input_channels_per_layer[i]
        x_now_group = data[:, start:end, indexes[start]]
        scale_factor = int(max_len/x_now_group.shape[-1])
        upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        upsampled_tensor = upsample(x_now_group)
        list.append(upsampled_tensor)
        start = end
    concatenated_tensor = torch.cat(list, dim=1) #b,c,l
    return concatenated_tensor

def down_sampling(raw_data,max_len,multi_sampling_rates,indexes):
    data = raw_data[:,:,:max_len]  #b,c,l
    common_elements = set(indexes[0])
    # 与其他列表的集合进行交集运算
    for l in indexes[1:]:
        common_elements = common_elements.intersection(l)
    sorted_list = sorted(common_elements)
    index = np.array(sorted_list)
    max_len = len(index)

    concatenated_tensor = data[:,:,index]
    return concatenated_tensor,max_len

def up_sampling_generator(data_path,max_len,multi_sampling_rates,indexes,args, augment=False):
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))

    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    # trainAll_dataset = torch.load(os.path.join(data_path, "train_all.pt"))

    train_dataset['samples'] = up_sampling(train_dataset['samples'],max_len,multi_sampling_rates,indexes)

    test_dataset['samples'] = up_sampling(test_dataset['samples'],max_len,multi_sampling_rates,indexes)

    train_dataset = Load_Dataset(train_dataset, args, augment)

    test_dataset = Load_Dataset(test_dataset, args, augment)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,shuffle=False, drop_last=False, num_workers=0)
    return train_loader, test_loader

def up_sampling_generator_part(data_path,max_len,multi_sampling_rates,indexes,part,args, augment=False):

    file_name = "train" + str(part) + ".pt"
    train_dataset = torch.load(os.path.join(data_path, file_name))
    train_dataset['samples'] = up_sampling(train_dataset['samples'], max_len, multi_sampling_rates, indexes)
    train_dataset = Load_Dataset(train_dataset, args, augment)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, num_workers=0)
    return train_loader

def down_sampling_generator(data_path,max_len,multi_sampling_rates,indexes,args, augment=False):
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    # trainAll_dataset = torch.load(os.path.join(data_path, "train_all.pt"))

    train_dataset['samples'],length = down_sampling(train_dataset['samples'],max_len,multi_sampling_rates,indexes)

    test_dataset['samples'],_ = down_sampling(test_dataset['samples'],max_len,multi_sampling_rates,indexes)

    train_dataset = Load_Dataset(train_dataset, args, augment)
    test_dataset = Load_Dataset(test_dataset, args, augment)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,shuffle=False, drop_last=False, num_workers=0)

    return train_loader, test_loader,length

def down_sampling_generator_part(data_path,max_len,multi_sampling_rates,indexes,part,args, augment=False):
    file_name = "train" + str(part) + ".pt"
    train_dataset = torch.load(os.path.join(data_path, file_name))
    train_dataset['samples'], _ = down_sampling(train_dataset['samples'],max_len,multi_sampling_rates,indexes)
    train_dataset = Load_Dataset(train_dataset, args, augment)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, num_workers=0)
    return train_loader
