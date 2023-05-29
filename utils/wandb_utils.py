import wandb

def init_wandb(name, project, config_dict):
    wandb.init(name=name, project=project)
    for key, value in config_dict.items():
        setattr(wandb.config, key, value)

def wandb_log_panel(log_dict, prefix = ''):
    temp_dict = {}
    for k in log_dict.keys():
        temp_dict[prefix+k] = log_dict[k]
    wandb.log(temp_dict)
    del temp_dict