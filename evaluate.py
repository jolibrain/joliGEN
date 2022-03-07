from options.evaluation_options import EvaluationOptions
import signal
from train import launch_training, signal_handler
import json

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler) #to really kill the process
    opt = EvaluationOptions().parse()   # get training options
    opt.train_compute_fid = True
    dataset_size_list = [int(size) for size in opt.eval_dataset_sizes.split(',')]
    dataset_size_list.sort(reverse=True)

    with open(opt.checkpoints_dir +"/" + opt.name + "/eval_results.json" , 'w+') as outfile:
        json.dump({}, outfile)
    
    for i,dataset_size in enumerate(dataset_size_list):
        print('-----------------%dth step over %d-----------------'%(i,len(dataset_size_list)))
        opt.data_max_dataset_size=dataset_size
        ###Trainings
        launch_training(opt)
