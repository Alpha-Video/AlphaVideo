from alphavideo.mot.TubeTK.TubeTK import TubeTK
from alphavideo.mot.TubeTK.config import __C


def tubeTK(num_class=1, pretrain=True):
    return TubeTK(arg=__C, num_classes=num_class, pretrained=pretrain)


if __name__=="__main__":
    # usage
    import torch

    # model
    model = tubeTK(pretrain=True).cuda()
    print(model)

    # input
    images = torch.zeros((1, 3, 8, 640, 896)).cuda()
    image_meta = [{'img_shape': [8, 896, 1152],
                   'value_range': 1,
                   'pad_percent': [1, 1]}]
    gt_tubes = [torch.tensor([[0, 0, 0.1, 0.1, 3, 2, 0, 0, 0.1, 0.1, -2, 0, 0, 0.1, 0.1]]).cuda()]
    gt_labels = [torch.ones((1, 1)).cuda()]

    # inference
    with torch.no_grad():
        results = model(images, image_meta, return_loss=False)
        print(results)

    # train
    results = model(images, image_meta, return_loss=True, gt_tubes=gt_tubes, gt_labels=gt_labels)
