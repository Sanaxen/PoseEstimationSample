SET WGET_EXE=.\wget\wget.exe
SET OPENPOSE_URL=http://posefs1.perception.cs.cmu.edu/OpenPose/models/
SET PROTOTXT_URL=https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose

%WGET_EXE% %OPENPOSE_URL%/pose/coco/pose_iter_440000.caffemodel
%WGET_EXE% %OPENPOSE_URL%/pose/mpi/pose_iter_160000.caffemodel
%WGET_EXE%  %PROTOTXT_URL%/mpi/pose_deploy_linevec_faster_4_stages.prototxt
%WGET_EXE%  %PROTOTXT_URL%/coco/pose_deploy_linevec.prototxt

