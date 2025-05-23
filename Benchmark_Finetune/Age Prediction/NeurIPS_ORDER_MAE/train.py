from data_loaders_4binover import get_loader
from logger import Logger
from sklearn import metrics as skmet
import argparse, datetime, json
import numpy as np
import os
import torch
import sys
import torch.nn as nn

from ordinal_loss import order_loss, euclidean_loss, order_loss_p
from mean_variance_loss import MeanVarianceLoss

from models import resnet

# Constants for MeanVarianceLoss
LAMBDA_1 = 0.05
LAMBDA_2 = 0.2
START_AGE = 10
END_AGE = 95

# Dictionary to store metric functions
metrics_f = {
    "accuracy": skmet.accuracy_score,
    "mae": skmet.mean_absolute_error
}

# Dictionary to store loss functions
loss_functions = {
    "ce": nn.CrossEntropyLoss(),
    "order": order_loss,
    "mv": MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE),
    "mse": nn.MSELoss(),
    "euc": euclidean_loss,
}

# Function to get the number of parameters in a model
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# Training function
def train(model, dataset, losses, l_d, opt, epoch, device, cls_num=2):
    
    model.train()
    data = iter(dataset)

    metrics, losses_dict = {}, {}

    for it in range(len(dataset)):
    #for it in range(20):
        try:
            x, y = next(data)
        except StopIteration:
            data = iter(dataset)
            x, y = next(data)

        x = x.to(device)
        y = y.to(device)

        if torch.isnan(x).any():
            print(f"Skipping batch at Iter {it} due to NaN in input")
            continue

        features, y_ = model(x)

        for loss in losses:
            if loss == "order" or loss == "euc":
                losses_dict[loss] = l_d*loss_functions[loss](features, y)
            elif loss == "mv":
                mean_loss, var_loss = loss_functions[loss](y_, y[:, 0])
                losses_dict[loss] = mean_loss + var_loss
            else:
                losses_dict[loss] = loss_functions[loss](y_, y[:, 0])

        closs = sum(losses_dict.values())

        opt.zero_grad()
        closs.backward()
        opt.step()

        for x in losses_dict:
            if x in metrics:
                metrics[x].append(losses_dict[x].item())
            else:
                metrics[x] = [losses_dict[x].item()]

        if "total_loss" in metrics:
            metrics["total_loss"].append(closs.item())
        else:
            metrics["total_loss"] = [closs.item()]

        log = f"Epoch {epoch+1}, Iter {it+1}/{len(dataset)}:"
        for m in sorted(metrics.keys()):
            log += f" {m} = {np.mean(metrics[m])}"

        print(log)

    return metrics

# Validation and testing function
def val_test(model, dataset, losses, l_d, epoch, device, cls_num=2, mode="val"):
    
    model.eval()
    data = iter(dataset)

    metrics, losses_dict = {}, {}

    gt = []
    pred = []

    with torch.no_grad():
        for it in range(len(dataset)):
            try:
                x, y = next(data)
            except:
                data = iter(dataset)
                x, y = next(data)

            x = x.to(device)
            y = y.to(device)

            features, y_ = model(x)

            for loss in losses:
                if loss == "order" or loss == "euc":
                    losses_dict[loss] = l_d*loss_functions[loss](features, y)
                elif loss == "mv":
                    mean_loss, var_loss = loss_functions[loss](y_, y[:, 0])
                    losses_dict[loss] = mean_loss + var_loss
                else:
                    losses_dict[loss] = loss_functions[loss](y_, y[:, 0])

            closs = sum(losses_dict.values())

            for x in losses_dict:
                if x in metrics:
                    metrics[x].append(losses_dict[x].item())
                else:
                    metrics[x] = [losses_dict[x].item()]

            if "total_loss" in metrics:
                metrics["total_loss"].append(closs.item())
            else:
                metrics["total_loss"] = [closs.item()]

            for x in y.cpu().numpy().squeeze():
                gt.append(x)
            for y in torch.argmax(y_, dim=1).cpu().numpy():
                pred.append(y)

        gt_arr = np.array(gt)
        pred_arr = np.array(pred)
        print(f'gt: {gt}')
        print(f'pred: {pred}')

        if mode == 'test':
            np.save(f'/opt/localdata/data/usr-envs/wenyi/ORDER_MAE/pred/true_epoch{epoch}.npy', gt_arr)
            np.save(f'/opt/localdata/data/usr-envs/wenyi/ORDER_MAE/pred/pred_epoch{epoch}.npy', pred_arr)


        for mf in metrics_f.keys():
            if mf in metrics:
                metrics[mf].append( metrics_f[mf](gt_arr, pred_arr) )
            else:
                metrics[mf] = [ metrics_f[mf](gt_arr, pred_arr) ]

        log = f"Testing on {mode.upper()} Dataset at Epoch {epoch+1}:"
        log += f" loss = {np.mean(metrics['total_loss'])}"
        log += f" accuracy = {np.mean(metrics['accuracy'])}"
        log += f" mae = {np.mean(metrics['mae'])}"

        print(log)

    return metrics

# Main function to parse arguments and run training and testing
def main():

    parser = argparse.ArgumentParser(description='3D Brain Age prediction')

    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='number of samples in each batch')

    # Data Parameters
    parser.add_argument('-d', '--dataset', default='data', type=str)
    parser.add_argument('--augmentation', action='store_true')

    # Losses used
    parser.add_argument('--losses', default='ce', type=str, nargs='+', metavar='BETA',
                        help='losses (default: Cross Entropy (ce), add others)')
    parser.add_argument('--ld', default=0.1, type=float, metavar='N',
                        help='l_d for order_loss')
    
    # Model Parameters
    parser.add_argument('--model_name', default="resnet18", type=str,
                        help='name of the classification model')
    parser.add_argument('--num_classes', default=86, type=int, metavar='N',
                        help='number of classes to predict')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')

    args = parser.parse_args()
    print(args)


    args.model_name = 'resnet34'
    args.batch_size = 8
    args.num_classes = 55
    args.losses = ['ce', 'order']
    args.ld = 0.1
    print(args) 
     
    # import sys
    os.chdir('/opt/localdata/data/usr-envs/wenyi/ORDER_MAE')

    # sys.path.insert('/opt/localdata/data/usr-envs/wenyi/ORDER')


    if "mse" in args.losses:
        args.num_classes = 1
    
    if args.batch_size < 2 and "order" in args.losses:
        print("Batch size should be at least 2 if you are using order loss!")
        exit()

    timestamp = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    training_folder = f"/opt/localdata/data/usr-envs/wenyi/ORDER_MAE/{args.dataset}_{args.model_name}_{timestamp}"

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
        os.makedirs(f"{training_folder}/weights")
    else:
        import sys
        sys.exit()

    with open(f'{training_folder}/params.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    logger = Logger(f'{training_folder}/logs')
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # Model selection based on model_name argument
    if args.model_name == "resnet10":
      model = resnet.generate_model(model_depth=10,
                                    n_classes=args.num_classes,
                                    n_input_channels=1,
                                    widen_factor=1)

    elif args.model_name == "resnet18":
      model = resnet.generate_model(model_depth=18,
                                    n_classes=args.num_classes,
                                    n_input_channels=1,
                                    widen_factor=1)

    elif args.model_name == "resnet34":
      model = resnet.generate_model(model_depth=34,
                                    n_classes=args.num_classes,
                                    n_input_channels=1,
                                    widen_factor=1)

    elif args.model_name == "resnet50":
      model = resnet.generate_model(model_depth=50,
                                    n_classes=args.num_classes,
                                    n_input_channels=1,
                                    widen_factor=1)

    elif args.model_name == "resnet101":
      model = resnet.generate_model(model_depth=101,
                                    n_classes=args.num_classes,
                                    n_input_channels=1,
                                    widen_factor=1)

    else:
      print(f"Invalid model_name: {args.model_name}!")
      exit()

    print("Training folder: ", training_folder)
    print("Model parameters: ", get_n_params(model))

    pretrained_mae_path = "/opt/localdata/data/usr-envs/wenyi/ORDER_MAE/MAE_train_model/model_final.pth"  # 🔁 Replace with actual file
    model.initialize_encoder_from_pretrained(pretrained_mae_path)


    # Load datasets
    dataset_train = get_loader(f'./{args.dataset}/HC/train', batch_size=args.batch_size, mode="train")
    dataset_val = get_loader(f'./{args.dataset}/HC/val', batch_size=args.batch_size, mode="val")
    dataset_test = get_loader(f'./{args.dataset}/HC/test', batch_size=args.batch_size, mode="test")

    # Optimizer selection based on opt argument
    if args.opt == "adam":
        opt = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay=1e-2)
    else:
      print(f"Invalid OPTIMIZER: {args.opt}!")
      exit()
    
    model.to(device)

    # Training loop
    for e in range(args.epochs):
        
      loss_metrics = train(model, dataset_train, args.losses, args.ld, opt, e, device, args.num_classes)
      
      for tag, value in loss_metrics.items():
          for cnt in range(len(value)):
              logger.scalar_summary("train/"+tag, value[cnt], e * len(loss_metrics[tag]) + cnt + 1)

      mpath = os.path.join(f"{training_folder}/weights", '{}.ckpt'.format(e+1))
      torch.save(model.state_dict(), mpath)
        
      val_loss_metrics = val_test(model, dataset_val, args.losses, args.ld, e, device, args.num_classes, "val")
      for tag, value in val_loss_metrics.items():
          logger.scalar_summary("val/"+tag, np.mean(value), (e+1))

      test_loss_metrics = val_test(model, dataset_test, args.losses, args.ld, e, device, args.num_classes, "test")
      for tag, value in test_loss_metrics.items():
          logger.scalar_summary("test/"+tag, np.mean(value), (e+1))


if __name__ == '__main__':
    main()
