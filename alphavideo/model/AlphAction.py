from alphavideo.action.AlphAction.ActionDetector import action_detector_res50, action_detector_res101

def alphaction_res50(num_class=80, pretrain=True):
    return action_detector_res50(num_class=num_class, pretrain=pretrain)

def alphaction_res101(num_class=80, pretrain=True):
    return action_detector_res101(num_class=num_class, pretrain=pretrain)

# usage

if __name__=="__main__":
    import torch

    # model
    model = alphaction_res101(pretrain=True).cuda()

    # online use, no future memory feature
    with torch.no_grad():
        for t in range(10):
            videos = torch.zeros((1, 3, 32, 256, 256)).cuda()
            video_metas = [dict(movie_id="tmp", timestamp=t)]
            box = [[[1, 1, 10, 10], [20, 30, 80, 100]]]
            objects = [[[50, 50, 60, 60]]]
            results = model(videos, box, video_metas, objects)
            print(results[0].get_field("scores"))

    # clear memory feature
    model.empty_pool()

    # offline use, use future memory feature
    with torch.no_grad():
        # extract features for each clip in the same long video.
        for t in range(10):
            videos = torch.zeros((1, 3, 32, 256, 256)).cuda()
            video_metas = [dict(movie_id="tmp", timestamp=t)]
            box = [[[1, 1, 10, 10], [20, 30, 80, 100]]]
            objects = [[[50, 50, 60, 60]]]
            model.store_feature(videos, box, video_metas, objects)
        device = videos.device

        # get the detection result of each clip.
        for t in range(10):
            video_metas = [dict(movie_id="tmp", timestamp=t)]
            results = model.get_prediction(video_metas=video_metas, device=device)
            print(results[0].get_field("scores"))

    # clear memory feature
    model.empty_pool()