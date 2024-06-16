import os
import sys
from time import time
import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

from datasets import dataset_factory
from datasets.augmentation import *
from datasets.graph import Graph
from evaluate import evaluate_fatigue

from common_fatigue import *
from utils import AverageMeter

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time()
    for idx, (points, labels) in enumerate(train_loader):
        data_time.update(time() - end)

        # points = torch.cat([points[0], points[1]], dim=0)
        if torch.cuda.is_available():
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=opt.use_amp):
            # compute loss
            logits = model(points)
            
            loss = criterion(logits, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        # print info
        if (idx + 1) % opt.log_interval == 0:
            print(
                f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t"
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"loss {losses.val:.3f} ({losses.avg:.3f})"
            )
            sys.stdout.flush()

    return losses.avg


def trainer(opt):
    opt = setup_environment(opt)
    graph = Graph("ntu")
    # Model & criterion
    model, model_args = get_model_resgcn(graph, opt)
    criterion = nn.BCELoss()

    print("# parameters: ", count_parameters(model))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, opt.gpus)

    if opt.cuda:
        model.cuda()
        criterion.cuda()
    # Dataset
    transform = transforms.Compose(
        [
            MirrorPoses(opt.mirror_probability),
            FlipSequence(opt.flip_probability),
            ShuffleSequence(opt.shuffle),
            PointNoise(std=opt.point_noise_std),
            JointNoise(std=opt.joint_noise_std),
            MultiInput(graph.connect_joint, opt.use_multi_branch),
            ToTensor()
        ],
    )

    dataset_class = dataset_factory(opt.dataset)
    dataset = dataset_class(
        opt.train_data_path,
        opt.view,
        transform=TwoNoiseTransform(transform),
    )
    best_acc = 0
    loss = 0
    # Tensorboard
    writer = SummaryWriter(log_dir=opt.tb_path)
    sample_input = torch.zeros(opt.batch_size, model_args["num_input"], model_args["num_channel"],
                            dataset.max_length, graph.num_node).cuda()
    writer.add_graph(model, input_to_model=sample_input)
    kfold = KFold(n_splits=5, shuffle=True)
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'Fold: {fold}')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(dataset, batch_size=opt.batch_size, 
            num_workers=opt.num_workers, pin_memory=True, sampler=train_subsampler)
        testloader = DataLoader(dataset, batch_size=opt.batch_size, 
            num_workers=opt.num_workers, pin_memory=True, sampler=test_subsampler)

        # Trainer
        optimizer, scheduler, scaler = get_trainer(model, opt, len(trainloader))

        # Load checkpoint or weights
        load_checkpoint(model, optimizer, scheduler, scaler, opt)

        for epoch in range(opt.start_epoch, opt.epochs + 1):
            # train for one epoch
            time1 = time()
            loss = train(
                trainloader, model, criterion, optimizer, scheduler, scaler, epoch, opt
            )

            time2 = time()
            print(f"epoch {epoch}, total time {time2 - time1:.2f}")

            # tensorboard logger
            writer.add_scalar("loss/train", loss, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

            # evaluation
            p, r, f1, acc = evaluate_fatigue(
                testloader, model, opt.evaluation_fn, use_flip=False
            )
            # writer.add_text("accuracy/validation", dataframe.to_markdown(), epoch)
            writer.add_scalar("precision", p, epoch)
            writer.add_scalar("recall", r, epoch)
            writer.add_scalar("f1", f1, epoch)
            writer.add_scalar("accuracy", acc, epoch)

            print(f"E: {epoch}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}, loss: {loss:.4f}")
            is_best = acc > best_acc
            if is_best:
                best_acc = acc

            if opt.tune:
                tune.report(accuracy=acc)

            if epoch % opt.save_interval == 0 or (is_best and epoch > opt.save_best_start * opt.epochs):
                save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{'best' if is_best else epoch}.pth")
                save_model(model, optimizer, scheduler, scaler, opt, opt.epochs, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, scheduler, scaler, opt, opt.epochs, save_file)

    log_hyperparameter(writer, opt, best_acc, loss)

    print(f"best accuracy: {best_acc*100:.2f}")


def _inject_config(config, opt):
    opt_new = {k: config[k] if k in config.keys() else v for k, v in vars(opt).items()}
    trainer(argparse.Namespace(**opt_new))


def tune_(opt):
    hyperband = HyperBandScheduler(metric="accuracy", mode="max")

    analysis = tune.run(
        _inject_config,
        config={},
        opt=opt,
        stop={"accuracy": 0.90, "training_iteration": 100},
        resources_per_trial={"gpu": 1},
        num_samples=10,
        scheduler=hyperband
    )

    print("Best config: ", analysis.get_best_config(metric="accuracy", mode="max"))

    df = analysis.results_df
    print(df)


def main():
    opt = parse_option()

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    opt.model_name = f"{date}_{opt.dataset}_{opt.network_name}" \
                     f"_lr_{opt.learning_rate}_decay_{opt.weight_decay}_bsz_{opt.batch_size}"

    if opt.exp_name:
        opt.model_name += "_" + opt.exp_name

    opt.model_path = f"../save/{opt.dataset}_models"
    opt.tb_path = f"../save/{opt.dataset}_tensorboard/{opt.model_name}"

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.evaluation_fn = None

    if opt.tune:
        tune_(opt)
    else:
        trainer(opt)


if __name__ == "__main__":
    main()
