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

* Parameters: cam_index 0, dataset_path /home/semester-thesis/RelisticRendering-dataset/, ref_img_index 1, blurred_img_index 2, n_images 20, initial_offset_pos 0.05, initial_offset_rot 0, output_file "results", sigma 0

* no changes
 

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

* changes: downscale by 2

`0,1,2,0,0.0234362,0.010823,0.0428206,1,0,0,-0,0.05,0,20,-0.0208886,-0.00249214,
-0.0412228,0.999997,-0.0020181,0.00139245,0.00087195,0.0462802,0.00520459,14,40.3145,1`

`0,1,2,0,0.0217368,0.000567419,0.0450243,1,0,0,0,0.05,0,20,-0.0209145,0.00127586,
-0.0450689,1,-0.000609736,0.000265314,0.000599421,0.0497016,0.0017905,30,134.118,1`

`0,1,2,0,0.0283446,0.0156478,-0.0381016,1,0,0,0,0.05,0,20,-0.0768047,0.0210969,
0.00874749,0.999212,-0.0301744,0.00538273,0.0252097,0.0801284,0.0793933,22,128.384,1`

`0,1,2,0,0.0351903,0.0223103,-0.0276387,1,0,0,-0,0.05,0,20,-0.07203,0.0225387,
0.00842621,0.999478,-0.0255065,0.00454813,0.0192918,0.0759429,0.0646159,33,177.87,1`

`0,1,2,0,0.0317983,0.0157217,0.0352377,1,0,0,0,0.05,0,20,-0.0256164,0.00094773,
-0.0313094,0.999979,-0.00525546,0.00191544,0.00341275,0.0404646,0.0131052,18,68.0318,1`

`0,1,2,0,0.0492496,0.00853881,0.0012532,1,0,0,0,0.05,0,20,-0.0681959,0.0385115,
-0.0302502,0.999631,-0.0236266,0.000147733,0.0134088,0.0839577,0.0543402,34,199.982,1`

`0,1,2,0,0.00988685,0.00165339,0.0489849,1,0,0,0,0.05,0,20,-0.00987959,-0.00160606,
-0.0489927,1,-1.00167e-05,6.4491e-06,1.23544e-05,0.0500047,3.43253e-05,13,31.1108,1`

`0,1,2,0,0.0299998,0.00803216,-0.0391854,1,0,0,0,0.05,0,20,-0.070291,0.0218145,
0.0235828,0.999562,-0.0235443,0.00306696,0.0176926,0.0772841,0.0592292,23,114.826,1`

`0,1,2,0,0.0406053,0.0283633,0.00683616,1,0,0,0,0.05,0,20,-0.0311222,0.00523851,
-0.00521981,0.99988,-0.0122392,0.00246554,0.00913618,0.0319888,0.0309429,28,140.08,1`

`0,1,2,0,0.00838475,0.00113555,-0.0492789,1,0,0,0,0.05,0,20,-0.0714214,0.0289269,
0.0142984,0.999677,-0.0160444,0.00590152,0.0188135,0.0783723,0.0508464,21,112.883,1`

* changes: downscale by 4

`0,1,2,0,0.000198409,3.34307e-05,-0.0499996,1,0,0,-0,0.05,0,20,-0.0930556,0.00185383,
0.0106793,0.999484,-0.0168742,0.00965881,0.0255862,0.0936847,0.0642819,29,40.1815,1`

`0,1,2,0,0.048107,0.0119612,0.00653083,1,0,0,0,0.05,0,20,-0.0557974,-0.00761671,
-0.0183004,0.999822,-0.016044,0.000150075,0.00996423,0.0592137,0.0377762,15,15.86,1`

`0,1,2,0,0.0421666,0.013428,-0.0232737,1,0,0,-0,0.05,0,20,-0.0957014,-0.0020042,
-0.0220613,0.999524,-0.0226905,-0.00206595,0.0208133,0.0982318,0.0617293,36,49.4118,1`

`0,1,2,0,0.0425236,0.000842045,0.0262875,1,0,0,0,0.05,0,20,-0.0427546,9.85006e-05,
-0.0265531,0.999999,-0.00111089,0.000185632,0.00103239,0.0503293,0.00305573,13,7.70392,1`

`0,1,2,0,0.0194494,0.000699501,-0.0460568,1,0,0,-0,0.05,0,20,-0.0995618,0.00577921,
-0.00319886,0.999517,-0.0164711,0.00538703,0.0258156,0.0997807,0.0621956,19,30.2129,1`

`0,1,2,0,0.00413855,0.00255189,0.049763,1,0,0,0,0.05,0,20,-0.0810289,-0.00536368,
-0.0925989,0.998632,-0.0189853,0.0135421,0.0467922,0.123163,0.10461,29,46.2105,1`

`0,1,2,0,0.0437346,0.0238421,-0.00434059,1,0,0,0,0.05,0,20,-0.0735009,-0.014201,
-0.0194122,0.999496,-0.0274319,-0.000984107,0.0159643,0.0773362,0.0635193,29,39.2607,1`

`0,1,2,0,0.0322683,0.0203981,-0.0322905,1,0,0,-0,0.05,0,20,-0.0788027,-0.0166948,
-0.0153666,0.999302,-0.0310383,-0.00479969,0.0202448,0.0820043,0.0747507,29,40.5444,1`

`0,1,2,0,0.0477883,0.0135976,-0.00560202,1,0,0,-0,0.05,0,20,-0.0907125,-0.0033598,
-0.021059,0.999568,-0.0228822,-0.00174737,0.0183743,0.0931854,0.0588052,29,36.4342,1`

`0,1,2,0,0.0209076,0.00086078,0.0454107,1,0,0,0,0.05,0,20,-0.0209481,-0.000844381,
-0.0454147,1,-1.54397e-05,1.99997e-06,3.34618e-05,0.0500203,7.38127e-05,16,10.8831,1`
