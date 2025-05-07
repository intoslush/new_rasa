from .build_dataloader import create_dataset, create_sampler, create_loader
from .sample import ValidIndexDistributedSampler
from utils.comm import get_rank,get_world_size
import ruamel.yaml as YAML 

# Dataset
def get_dataloder(args):
    print("Creating retrieval dataset")
    yaml = YAML.YAML(typ='rt') 
    config = yaml.load(open(args.config, 'r')) 
    train_dataset, val_dataset, test_dataset = create_dataset('ps', config)
   
    if args.distributed:
        sampler = ValidIndexDistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank())
    else:
        sampler = None

    samplers = [sampler, None, None]
    #dataloader在这,然后logevery是直接迭代这个
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                            batch_size=[config['batch_size_train']] + [
                                                                config['batch_size_test']] * 2,
                                                            num_workers=[4, 4, 4],
                                                            is_trains=[True, False, False],
                                                            collate_fns=[None, None, None])
    cluster_lodaer = create_loader([train_dataset], [None],
                                                            batch_size=[config['batch_size_train']],
                                                            num_workers=[4],
                                                            is_trains=[False],
                                                            collate_fns=[None])[0]

    return train_loader, val_loader, test_loader, cluster_lodaer