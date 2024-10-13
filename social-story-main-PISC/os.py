import os
import argparse

def reset(dir_result):
    for root,dirs,files in os.walk(os.path.join(dir_result,'caption','caption_sam')):
        number_list = []
        for file in files:
            number = int(file.split('.')[0].split('_')[-1])
            number_list.append(number)
    
    maxnum = max(number_list)
    os.system('bash main.sh '+str(maxnum-1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_result',default='result_try',type=str)
    parser.add_argument('--index',default=100,type=int)
    args = parser.parse_args()

    os.system('bash main.sh')
    idx = args.index
    while idx > 0:
        idx = idx - 1
        reset(args.dir_result)


