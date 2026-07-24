from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_MIRRORS = {
    "official": "",
    "sjtu": "https://scidata.sjtu.edu.cn/records/p4t8m-rbe26/files/cifar-10-python.tar.gz?download=1",
    "oneflow": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz",
    "baidu": "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz",
    "brainchip": "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
}


def resolve_download_url(mirror="", download_url=""):
    if download_url:
        return download_url
    if mirror not in CIFAR10_MIRRORS:
        names = ", ".join(sorted(CIFAR10_MIRRORS))
        raise ValueError(f"unknown mirror '{mirror}', choose from: {names}")
    return CIFAR10_MIRRORS[mirror]


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=14),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, test_transform


def build_loaders(args):
    download_url = resolve_download_url(
        getattr(args, "mirror", "official"), getattr(args, "download_url", "")
    )
    if download_url:
        datasets.CIFAR10.url = download_url

    train_transform, test_transform = build_transforms()
    train_set = datasets.CIFAR10(
        root=args.data_dir, train=True, transform=train_transform, download=True
    )
    test_set = datasets.CIFAR10(
        root=args.data_dir, train=False, transform=test_transform, download=True
    )

    if args.limit_train_samples > 0:
        train_set = Subset(train_set, range(args.limit_train_samples))
    if args.limit_test_samples > 0:
        test_set = Subset(test_set, range(args.limit_test_samples))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return train_loader, test_loader


def download_cifar10(data_dir, mirror="official", download_url=""):
    download_url = resolve_download_url(mirror, download_url)
    if download_url:
        datasets.CIFAR10.url = download_url

    train_transform, test_transform = build_transforms()
    datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)


def build_predict_transform():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
