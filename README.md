Work Overview - Semester Thesis from Johann Diep (jdiep@student.ethz.ch)

Docker commands 
------------

### Accessing root

```bash
sudo -s
```

### Stop all containers
```bash
docker stop $(docker ps -a -q)
```

### Delete all containers
```bash
docker rm $(docker ps -a -q)
```

### Delete all images
```bash
docker rmi $(docker images -q)
```

### Building image
```bash
docker build -t st-ubuntu .
```

### Run image
```bash
docker run -it st-ubuntu
```

### Show all images
```bash
docker images
```

### Show all containers
```bash
docker ps -a
```

### Copy file to host machine
```bash
docker cp cocky_vaughan:/home/semester-thesis/file /home/johann
```

### After running bash
```bash
apt-get install libmrpt-dev
git clone https://gitlab.com/jdiep/semester-thesis -b 1-coarse-to-fine-implementation
```

Hyperparameters chosen so far
------------

* Number of poses to generate synthetical image: N_poses (must be defined, tested between 3-20)
* times at pose P_t: t_t (timestamp at closing shutter, given in the dataset)
* Exposure time: t_exp (must be given by the camera setup)
* Pose at the end: P_t <-> Initialization P_tilde -> Initialization offset 
* Pixel in the resulting blurred image are set to be non-empty only if 1/5 of the reprojected poses has information
* Penalizing empty pixels: a =5
* Cost function: C(P_t)
* Distance between two consecutive images: 26.7cm (std: 2.7cm)
* Distance traveled during open-shutter time: 10.6cm (std: 1.1cm)
* Gradient of intensity: nabla_I (Sobel filter)
* Scaling factor: sigma_0 (tuning such that pertubuation is not higher than 2, 0.02-0.30)
* Decay factor: alpha (set to 1x10^-10)



Implementation
------------

* Coarse-to-fine image resolution: Implementation of the coarse-to-fine method to run the minimalization step more efficiently. 

Results
------------

* no changes: cam_index 0, dataset_path /home/semester-thesis/RelisticRendering-dataset/, ref_img_index 1, blurred_img_index 2, n_images 20, initial_offset_pos 0.05, initial_offset_rot 0, output_file "results", sigma 0
 

`0,1,2,0,0.0397207,0.0267231,-0.0144272,1,0,0,0,0.05,0,20,-0.00707484,0.00901413,
0.00266361,0.999998,0.000429444,0.00177906,0.000122679,0.0117645,0.00366853,22,592.71,1`

`0,1,2,0,0.00751685,0.00163148,0.0494048,1,0,0,-0,0.05,0,20,-0.00667034,0.012418,
0.00532308,0.999998,0.00114136,0.0016039,0.000233644,0.0150677,0.00396474,32,1043.91,1`

`0,1,2,0,0.000535027,5.11745e-05,0.0499971,1,0,0,0,0.05,0,20,-0.00657229,0.0124662,
0.00537965,0.999998,0.00115143,0.00158353,0.0002368,0.0150845,0.00394433,44,1238.4,1`

`0,1,2,0,0.0249302,0.00422317,-0.0431353,1,0,0,-0,0.05,0,20,-0.00373951,0.00591205,
0.000845028,0.999999,-0.000275137,0.00122958,0.000178769,0.0070463,0.00254522,15,248.163,1`

`0,1,2,0,0.0463265,0.0176928,-0.00638919,1,0,0,-0,0.05,0,20,-0.00686643,0.00885003,
0.00259292,0.999998,0.000387634,0.00173999,0.000133206,0.0114976,0.00357523,31,806.284,1`

`0,1,2,0,0.0349764,0.0119813,0.0336616,1,0,0,-0,0.05,0,20,-0.0531292,0.0550532,
0.0110991,0.99858,-0.0460408,-0.00569135,0.0262071,0.0773095,0.106614,49,1416.27,1`

`0,1,2,0,0.0381867,0.00280302,0.0321546,1,0,0,0,0.05,0,20,-0.0541571,0.0555998,
0.0120172,0.998724,-0.0439994,-0.00608224,0.0240388,0.0785413,0.101054,46,1349.56,1`

`0,1,2,0,0.0310849,0.0160442,0.0357256,1,0,0,0,0.05,0,20,-0.00651511,0.00835556,
-0.0234161,0.999984,-0.0047696,-0.00258218,-0.0016655,0.0257017,0.0113474,11,146.015,1`

`0,1,2,0,0.0406404,0.0211414,0.020035,1,0,0,0,0.05,0,20,-0.0107441,0.00412692,
-0.00447648,0.999983,-0.00457544,-0.00347171,-0.00114682,0.0123493,0.0117138,19,302.991,1`

`0,1,2,0,0.0430998,0.0151478,0.0203213,1,0,0,0,0.05,0,20,-0.0154119,0.00957277,
-0.00993709,0.999982,-0.0045265,-0.00351292,-0.00170393,0.020686,0.0119555,24,411.968,1`
