## [Adapose](https://www.ijcai.org/proceedings/2021/184) as a postprocessor

Since the release of AdaPose in 2021(not to be mistaken with [the adapose that came out 2 years later](https://arxiv.org/abs/2309.16964)) there have been numerous gains in the field of *wholebody* pose detection, specifically [RTMPose](https://arxiv.org/abs/2303.07399)([GH](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)). 
Adapose is a great candidate for real time RGB-D fusion because it is both lightweight and utilizes 2D pose priors to build it's sampling areas. Using known camera intrinsics we can construct a network that wholly relies on the RGB based RTMPose to determine the epipolar line that the keypoint lies on, and merely have adapose output a Z value.
