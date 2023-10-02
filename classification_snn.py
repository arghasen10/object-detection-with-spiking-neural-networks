from os.path import join
import sys
import argparse
import torch
torch.manual_seed(0)
from torch.utils.data import DataLoader
from datasets.classification_datasets import NCARSClassificationDataset, GEN1ClassificationDataset
from snntorch import surrogate
from snntorch import functional as SF
from models.vgg_snn import VGG16
from models.densenet_snn import DenseNet
from models.mobilenet_snn import MobileNetV1


def calculate_accuracy(model, test_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    
    acc_hist = []
    with torch.no_grad():
        for data, targets in test_dataloader:
            data = data.permute(1, 0, 2, 3, 4).to(device)
            targets = targets.to(device)
            
            spk_rec = model(data.float())
            acc = SF.accuracy_rate(spk_rec, targets)            
            acc_hist.append(acc)

    accuracy = 100 * torch.Tensor(acc_hist).mean()
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Classify event dataset')
    parser.add_argument('-device', default=0, type=int, help='device')
    parser.add_argument('-precision', default=16, type=int, help='whether to use AMP {16, 32, 64}')

    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
    parser.add_argument('-tbin', default=2, type=int, help='number of micro time bins')
    parser.add_argument('-image_shape', default=(304,240), type=tuple, help='spatial resolution of events')

    parser.add_argument('-dataset', default='ncars', type=str, help='dataset used {NCARS, GEN1}')
    parser.add_argument('-path', default='datasets', type=str, help='dataset used. {NCARS, GEN1}')
    parser.add_argument('-undersample_cars_percent', default='0.24', type=float, help=
                        'Undersample cars in Prophesse GEN1 Classification by using only x percent of cars.')

    parser.add_argument('-model', default='vgg-16', type=str, help='model used {squeezenet-v, vgg-v, mobilenet-v, densenet-v}')
    parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-lr', default=5e-3, type=float, help='learning rate used')
    parser.add_argument('-epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-test', action='store_true', help='whether to test the model')

    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    parser.add_argument('-comet_api', default=None, type=str, help='api key for Comet Logger')

    args = parser.parse_args()
    print(args)

    if args.dataset == "ncars":
        dataset = NCARSClassificationDataset
    elif args.dataset == "gen1":
        dataset = GEN1ClassificationDataset
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

    train_dataset = dataset(args, mode="train")
    test_dataset = dataset(args, mode="test")

    train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.b, num_workers=8)
    #event_tensor, target = next(iter(train_dataloader))
    #print(event_tensor.shape)           # 64 batch, 5 time steps, 4 timebins, 64, 64 (H, W)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model == "vgg-16":
        net = VGG16(spike_grad = surrogate.atan(), beta = 0.9, num_classes=2).to(device)
    elif args.model == "densenet":
        net = DenseNet(spike_grad = surrogate.atan(), num_classes=2)
    elif args.model == "mobilenet":
        net = MobileNetV1(spike_grad = surrogate.atan(), num_classes=2)
        
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    #loss_fn = nn.CrossEntropyLoss()

    if args.pretrained is not None:
        net.load_state_dict(torch.load(args.pretrained))
        test_acc = calculate_accuracy(net, test_dataloader=test_dataloader, device=device)
        print(f'Test Accuracy: {test_acc}')
        return

    num_epochs = args.epochs

    loss_hist = []
    acc_hist = []
    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(train_dataloader)):
            data = data.permute(1, 0, 2, 3, 4).to(device)
            targets = targets.to(device)
            net.train()

            spk_rec = net(data.float())
            loss_val = torch.zeros((1), dtype=torch.float, device=device)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
            print('Inferred: ', spk_rec.sum(dim=0).max(1)[-1])
            print('Actual: ', targets)
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

    test_acc = calculate_accuracy(net, test_dataloader=test_dataloader, device=device)
    print(f'Test Accuracy: {test_acc}')

    if args.save_ckpt:
        filepath = f'saved_model/{args.model}_{args.epochs}_{args.T}_{args.b}.pth'
        torch.save(net.state_dict(), filepath)

        with open(f'saved_model/{args.model}_{args.epochs}_{args.T}_{args.b}_result.txt', 'w') as file:
            file.write(f'Test Accuracy: {test_acc}')



if __name__ == '__main__':
    main()
