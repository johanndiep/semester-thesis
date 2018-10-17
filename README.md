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
docker cp container_name:/home/semester-thesis/file /home/johann
```

### After running bash
```bash
apt-get install libmrpt-dev
git clone https://gitlab.com/jdiep/semester-thesis -b 1-coarse-to-fine-implementation
```

### Run multiple terminal of same container
```bash
docker exec -it container_name bash
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

* scale 1 (ground truth: -0.821577 1.31002 0.911207 0.951512 0.0225991 0.0716038 -0.298306)

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.043773,0.0191741,-0.0147065,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.0558453,0.0571309,0.0125707,0.999419,-0.031711,-0.00020714,0.0125264, solved_pose: -0.858742,1.26958,0.970572,0.305682,-0.34962,0.648962,-0.602639, err_dist: 0.0808743, err_rot_angle: 0.0682054, num_iterations: 24, total_time: 611.671, convergence: 1`

* sequential scale 4 -> 2 -> 1

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0485365,0.0117561,-0.00244991,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.0922032,-0.00303057,-0.0224219,0.999494,-0.0228472,-0.00212972,0.0220155, solved_pose: -0.856132,1.2216,0.909906,0.295818,-0.35759,0.651383,-0.600268, err_dist: 0.0949387, err_rot_angle: 0.0636098, num_iterations: 38, total_time: 49.7805, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0328039,0.0216427,0.0309111,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.0253902,-0.00341289,-0.0266993,0.999983,-0.0048051,0.00198077,0.00274948, solved_pose: -0.814612,1.27436,0.9042,0.303525,-0.35264,0.667659,-0.581162, err_dist: 0.0370022, err_rot_angle: 0.0117597, num_iterations: 23, total_time: 82.1145, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0202408,0.0136225,-0.0436433,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.0654652,0.0568786,0.0145086,0.999219,-0.0347688,0.000842579,0.0187704, solved_pose: -0.865533,1.26343,0.971442,0.303793,-0.353428,0.644551,-0.606096, err_dist: 0.0879282, err_rot_angle: 0.0790626, num_iterations: 22, total_time: 564.212, convergence: 1`

!!SO FAR WRONG FOCAL LENGTH, AND WRONG SCALING, WRONG RESULTS FOR  TESTS!!

* no changes, 0.05 offset

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0334174,0.0167196,-0.0332225,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00679383,0.00860116,0.00240611,0.999998,0.000333826,0.0017321,0.000131998, solved_pose: -0.826332,1.30504,0.920069,0.303071,-0.352295,0.671633,-0.577014, err_dist: 0.0112217, err_rot_angle: 0.00353782, num_iterations: 35, total_time: 833.712, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0378066,0.000805966,-0.0327111,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00672953,0.00813967,0.00206624,0.999998,0.000235673,0.0017312,0.000121692, solved_pose: -0.826078,1.30494,0.919561,0.303111,-0.352258,0.67158,-0.577077, err_dist: 0.0107615, err_rot_angle: 0.00350281, num_iterations: 26, total_time: 546.306, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0099336,0.00398862,0.0488407,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00646679,0.0118629,0.00496488,0.999998,0.00102046,0.00157693,0.000232886, solved_pose: -0.827811,1.30646,0.923684,0.302667,-0.352481,0.672041,-0.576638, err_dist: 0.0143943, err_rot_angle: 0.00378539, num_iterations: 49, total_time: 1532.78, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0281217,0.0061118,-0.0408879,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00502048,0.00653784,0.000991311,0.999999,-0.000107243,0.00148992,0.0001064, solved_pose: -0.824435,1.30589,0.917817,0.303078,-0.352005,0.67146,-0.577388, err_dist: 0.00830248, err_rot_angle: 0.00299511, num_iterations: 12, total_time: 208.65, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0199046,0.0106566,-0.0446122,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00189276,0.00553675,0.00071175,0.999999,-0.000388014,0.000892157,0.000263516, solved_pose: -0.822555,1.30839,0.916788,0.302684,-0.351681,0.671424,-0.577835, err_dist: 0.00589447, err_rot_angle: 0.00201588, num_iterations: 12, total_time: 210.029, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0490526,0.0039239,-0.00885686,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00723637,0.00689242,0.000980642,0.999998,-1.93338e-05,0.00187156,6.7908e-05, solved_pose: -0.825638,1.30402,0.918163,0.303325,-0.352226,0.671409,-0.577183, err_dist: 0.0100415, err_rot_angle: 0.00374578, num_iterations: 24, total_time: 546.459, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0431051,0.0244897,-0.00649682,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00694698,0.00892628,0.00263219,0.999998,0.000404745,0.00175578,0.000130998, solved_pose: -0.826562,1.30501,0.920424,0.303062,-0.35233,0.671667,-0.576958, err_dist: 0.0116132, err_rot_angle: 0.00361316, num_iterations: 33, total_time: 874.437, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.047948,0.0121674,0.00727637,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00674066,0.00813401,0.00206618,0.999998,0.000235142,0.00173343,0.000121248, solved_pose: -0.826084,1.30494,0.919555,0.303113,-0.352259,0.671579,-0.577077, err_dist: 0.0107642, err_rot_angle: 0.00350701, num_iterations: 31, total_time: 880.955, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0328191,0.0216654,0.0308791,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00656111,0.00945797,0.00312365,0.999998,0.000509087,0.00166027,0.000169539, solved_pose: -0.826675,1.30556,0.921025,0.302939,-0.352332,0.671743,-0.576933, err_dist: 0.0119272, err_rot_angle: 0.00348964, num_iterations: 46, total_time: 1355.44, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0461486,0.0152148,-0.0117822,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00693794,0.00890833,0.00260991,0.999998,0.000401234,0.00175459,0.000130649, solved_pose: -0.826541,1.30501,0.920403,0.303063,-0.352328,0.671665,-0.576961, err_dist: 0.011589, err_rot_angle: 0.00360924, num_iterations: 33, total_time: 872.536, convergence: 1`

* changes: downscale by 2, 0.05 offset

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0470915,0.00835264,-0.0145817,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00841189,0.00589851,-0.000912055,0.999998,-0.000285264,0.00194114,0.000118669, solved_pose: -0.824888,1.30208,0.916898,0.303436,-0.35222,0.671216,-0.577353, err_dist: 0.0103143, err_rot_angle: 0.00393116, num_iterations: 27, total_time: 151.438, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0304443,0.0157417,-0.0364052,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00708334,0.0066404,-5.40543e-05,0.999999,-0.000146878,0.00165529,0.000200155, solved_pose: -0.824742,1.30359,0.917762,0.303149,-0.352152,0.671355,-0.577385, err_dist: 0.00970935, err_rot_angle: 0.00334761, num_iterations: 28, total_time: 152.544, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0425381,0.0238222,0.0110911,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00723112,0.00402986,-0.00191508,0.999998,-0.000688092,0.00173183,9.93381e-05, solved_pose: -0.823635,1.30265,0.914904,0.303448,-0.351964,0.671054,-0.577691, err_dist: 0.00849685, err_rot_angle: 0.00373234, num_iterations: 26, total_time: 194.558, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.038288,0.0033328,0.0319831,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00748178,0.00689475,6.32142e-05,0.999999,-9.81348e-05,0.00171433,0.000187039, solved_pose: -0.825032,1.30331,0.91803,0.303179,-0.352192,0.671369,-0.577327, err_dist: 0.0101744, err_rot_angle: 0.00345458, num_iterations: 15, total_time: 87.1161, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0283137,0.013855,-0.038812,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00669841,0.00657465,-0.000194931,0.999999,-0.000158619,0.00161169,0.00020633, solved_pose: -0.824418,1.30384,0.917676,0.30312,-0.352127,0.671359,-0.57741, err_dist: 0.0093879, err_rot_angle: 0.00326514, num_iterations: 28, total_time: 128.404, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0384761,0.0196042,0.0252044,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00801606,0.00510946,-0.00133433,0.999998,-0.000454179,0.00188364,0.000103864, solved_pose: -0.824418,1.30224,0.916056,0.303465,-0.352126,0.671141,-0.577482, err_dist: 0.00959918, err_rot_angle: 0.00388081, num_iterations: 21, total_time: 122.679, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0337374,0.00504514,-0.0365559,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00632858,0.00595527,-0.00050714,0.999999,-0.000271668,0.00163176,0.000185475, solved_pose: -0.824031,1.30402,0.917017,0.303185,-0.35209,0.671295,-0.577473, err_dist: 0.00870478, err_rot_angle: 0.00332918, num_iterations: 12, total_time: 52.4495, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0192112,0.00811468,0.0454432,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00698784,0.00607946,-0.000435344,0.999999,-0.000272542,0.00162129,0.000177092, solved_pose: -0.824449,1.30351,0.917151,0.303183,-0.352078,0.6713,-0.577474, err_dist: 0.0092725, err_rot_angle: 0.0033071, num_iterations: 24, total_time: 156.758, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.00715293,0.00455599,0.0492755,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00718506,0.00682705,7.33106e-05,0.999999,-0.000114563,0.00163802,0.000205893, solved_pose: -0.824881,1.30357,0.917966,0.303122,-0.352155,0.671376,-0.577371, err_dist: 0.00991157, err_rot_angle: 0.00330976, num_iterations: 31, total_time: 195.339, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0378126,0.0150821,0.0290299,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00919391,0.00474822,-0.00189469,0.999998,-0.000502044,0.00214386,2.09474e-05, solved_pose: -0.824672,1.30098,0.915612,0.303705,-0.352206,0.671064,-0.577397, err_dist: 0.0105197, err_rot_angle: 0.00440393, num_iterations: 21, total_time: 120.265, convergence: 1`

* changes: downscale by 4, 0.05 offset

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.00647739,0.00135404,-0.0495602,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: 0.0064538,-0.00120418,-0.00259732,0.999997,-0.00218583,-0.000651947,0.000962007, solved_pose: -0.815924,1.31394,0.909626,0.301873,-0.350714,0.670605,-0.579794, err_dist: 0.00706029, err_rot_angle: 0.0049511, num_iterations: 23, total_time: 46.3568, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.040143,0.0269638,-0.0127078,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00984086,0.00117709,-0.00471734,0.999996,-0.00135324,0.00248164,0.00016612, solved_pose: -0.823169,1.29917,0.911671,0.304146,-0.352241,0.670418,-0.577894, err_dist: 0.0109764, err_rot_angle: 0.005663, num_iterations: 25, total_time: 56.1012, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0388778,0.019131,0.0249503,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00699983,0.0011512,-0.00442418,0.999997,-0.00140389,0.00192721,0.000342331, solved_pose: -0.821809,1.30168,0.911694,0.30369,-0.352024,0.670495,-0.578176, err_dist: 0.0083604, err_rot_angle: 0.00481757, num_iterations: 29, total_time: 66.2819, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0280811,0.0174566,-0.0375063,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00531219,0.00240272,-0.00262178,0.999998,-0.0012028,0.00152468,0.000508517, solved_pose: -0.822171,1.30397,0.913199,0.303253,-0.351964,0.670675,-0.578233, err_dist: 0.00639267, err_rot_angle: 0.00401497, num_iterations: 26, total_time: 49.7497, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0223518,0.00620232,0.0442936,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00620272,0.00271205,-0.00282741,0.999998,-0.00107845,0.00172786,0.000394657, solved_pose: -0.822469,1.3031,0.913472,0.303412,-0.352042,0.670726,-0.578044, err_dist: 0.00733643, err_rot_angle: 0.00414936, num_iterations: 23, total_time: 54.7128, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0324664,0.0165511,-0.0342344,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00590743,0.00248786,-0.00286441,0.999998,-0.00115004,0.00165436,0.000483263, solved_pose: -0.822299,1.30334,0.913246,0.303336,-0.352038,0.670675,-0.578144, err_dist: 0.00702083, err_rot_angle: 0.00414394, num_iterations: 31, total_time: 66.9588, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0432076,0.00443129,-0.0247682,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00616185,0.00225909,-0.00301166,0.999998,-0.0011845,0.00172113,0.00041406, solved_pose: -0.822351,1.30307,0.912998,0.303433,-0.35202,0.670659,-0.578123, err_dist: 0.00722094, err_rot_angle: 0.00425994, num_iterations: 19, total_time: 44.0659, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0177438,0.00271162,-0.046667,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: 0.00285119,0.00275407,-0.000213736,0.999999,-0.00127249,-0.000130264,0.000937565, solved_pose: -0.81944,1.31201,0.913895,0.301918,-0.351275,0.670985,-0.578991, err_dist: 0.00396987, err_rot_angle: 0.0031719, num_iterations: 18, total_time: 28.7691, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0384859,0.0274634,0.0162666,1,0,0,0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00879424,0.00166021,-0.00391849,0.999997,-0.00128208,0.0022381,0.000177234, solved_pose: -0.823173,1.30044,0.912267,0.303952,-0.352129,0.670529,-0.577935, err_dist: 0.00976983, err_rot_angle: 0.00517079, num_iterations: 15, total_time: 37.9761, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0325599,0.0179043,0.0334558,1,0,0,-0, initial_offset_dist: 0.05, initial_offset_rot_angle: 0, n_images: 20, err: -0.00654097,0.00263075,-0.00276536,0.999998,-0.00108038,0.00180147,0.000351306, solved_pose: -0.822721,1.30286,0.9134,0.303487,-0.352055,0.670717,-0.578006, err_dist: 0.00757313, err_rot_angle: 0.00425955, num_iterations: 27, total_time: 69.5214, convergence: 1`

* no changes, 0.1 offset

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0991166,0.0116075,0.00641651,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00678689,0.00790041,0.00187331,0.999998,0.000187995,0.00175183,0.000112055, solved_pose: -0.825983,1.30481,0.919295,0.303147,-0.352249,0.67155,-0.577099, err_dist: 0.0105824, err_rot_angle: 0.0035309, num_iterations: 28, total_time: 971.171, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.037019,0.00088411,-0.0928914,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00294321,0.00107657,-0.00260205,0.999998,-0.00133905,0.00113468,0.000181452, solved_pose: -0.821005,1.30604,0.911888,0.303228,-0.351478,0.670829,-0.578363, err_dist: 0.00407334, err_rot_angle: 0.00352901, num_iterations: 15, total_time: 456.257, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0541406,0.0385711,0.0747065,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00775571,0.000793006,-0.00418838,0.999997,-0.00130211,0.0021451,-0.000111503, solved_pose: -0.822471,1.30122,0.911366,0.304063,-0.351876,0.670647,-0.577894, err_dist: 0.00884999, err_rot_angle: 0.0050237, num_iterations: 44, total_time: 1441.45, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0973894,0.00390044,0.0223629,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00712269,0.00755743,0.00154007,0.999998,0.000122814,0.00182865,8.67808e-05, solved_pose: -0.825945,1.30437,0.918905,0.303236,-0.352257,0.671498,-0.577108, err_dist: 0.0104985, err_rot_angle: 0.00366964, num_iterations: 30, total_time: 937.627, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0306996,0.000481817,-0.0951698,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00212786,0.001294,-0.00222447,0.999999,-0.00130784,0.000957689,0.000227607, solved_pose: -0.820823,1.30691,0.912159,0.303072,-0.351416,0.670885,-0.578418, err_dist: 0.00333924, err_rot_angle: 0.00327379, num_iterations: 22, total_time: 503.17, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0837611,0.0147415,-0.0526,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00742609,0.00359981,-0.00166615,0.999998,-0.000716357,0.00199928,-2.88592e-05, solved_pose: -0.823998,1.30267,0.914513,0.303712,-0.352024,0.671001,-0.577577, err_dist: 0.00841912, err_rot_angle: 0.00424787, num_iterations: 35, total_time: 1033, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0446588,0.00409457,-0.0893802,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00299492,0.00168243,-0.00212422,0.999999,-0.00121042,0.00113022,0.000195277, solved_pose: -0.82135,1.30622,0.912557,0.303172,-0.351524,0.6709,-0.578283, err_dist: 0.00403887, err_rot_angle: 0.00333505, num_iterations: 14, total_time: 307.253, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0990037,0.0124128,-0.00664684,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00752054,0.0067575,0.000779833,0.999998,-4.09e-05,0.00193153,5.11398e-05, solved_pose: -0.825652,1.30369,0.917999,0.303383,-0.352243,0.671384,-0.577171, err_dist: 0.0101405, err_rot_angle: 0.00386529, num_iterations: 24, total_time: 672.702, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0744459,0.00554032,-0.0665365,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00822417,0.0022404,-0.00335627,0.999997,-0.00102087,0.0021144,9.50864e-05, solved_pose: -0.823239,1.30118,0.912921,0.303824,-0.352082,0.670747,-0.577779, err_dist: 0.00916084, err_rot_angle: 0.00469976, num_iterations: 23, total_time: 460.859, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.00503869,0.00208941,0.0998511,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00670702,0.00971049,0.00328118,0.999998,0.000566173,0.00168442,0.000168881, solved_pose: -0.826854,1.30551,0.921298,0.302936,-0.352363,0.671769,-0.576886, err_dist: 0.0122492, err_rot_angle: 0.00357007, num_iterations: 44, total_time: 1087.28, convergence: 1`

* changes: downscale by 2, 0.1 offset

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0547149,0.0162017,-0.0821206,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00555303,0.000544794,-0.0040048,0.999998,-0.00142664,0.00154448,0.00013303, solved_pose: -0.821408,1.30315,0.911155,0.303562,-0.351656,0.670672,-0.578263, err_dist: 0.00686814, err_rot_angle: 0.00421351, num_iterations: 12, total_time: 70.0803, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.065978,0.00683717,-0.0748342,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00850133,0.00320345,-0.00277985,0.999998,-0.000835504,0.00203441,8.67204e-05, solved_pose: -0.823749,1.30119,0.913957,0.30371,-0.352086,0.670881,-0.57768, err_dist: 0.00950064, err_rot_angle: 0.00440201, num_iterations: 12, total_time: 81.3961, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0820308,0.0568091,-0.00660846,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00823128,-0.000260117,-0.00540718,0.999997,-0.00154526,0.00210379,-0.000127485, solved_pose: -0.821875,1.30023,0.910147,0.30413,-0.351767,0.670525,-0.578067, err_dist: 0.00985186, err_rot_angle: 0.00522687, num_iterations: 34, total_time: 265.441, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0946145,0.0314783,0.00756424,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.0076823,0.00267463,-0.00298387,0.999998,-0.000954504,0.00187272,2.25471e-05, solved_pose: -0.823186,1.30179,0.913405,0.303681,-0.351914,0.670884,-0.577797, err_dist: 0.00866457, err_rot_angle: 0.00420412, num_iterations: 26, total_time: 218.825, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0833942,0.0143881,-0.0532765,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.0079431,0.00175976,-0.00375395,0.999997,-0.00114347,0.00196197,-2.5223e-05, solved_pose: -0.822817,1.30122,0.912387,0.303834,-0.351876,0.670764,-0.577878, err_dist: 0.00896001, err_rot_angle: 0.00454202, num_iterations: 33, total_time: 224.132, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0950848,0.0301693,0.00697826,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00768845,0.00265343,-0.00300025,0.999998,-0.000957203,0.00187497,2.11786e-05, solved_pose: -0.823179,1.30178,0.913382,0.303684,-0.351913,0.670882,-0.577798, err_dist: 0.00866917, err_rot_angle: 0.00421056, num_iterations: 29, total_time: 198.743, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0863817,0.0257864,0.0432812,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00780067,0.00259517,-0.00311153,0.999998,-0.000972561,0.00189677,1.79985e-05, solved_pose: -0.823159,1.30163,0.913307,0.303706,-0.351919,0.670868,-0.577799, err_dist: 0.00879016, err_rot_angle: 0.00426331, num_iterations: 28, total_time: 177.788, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.083123,0.000463836,0.055591,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00863008,0.00361687,-0.00264885,0.999998,-0.000745213,0.00206637,8.65926e-06, solved_pose: -0.823878,1.30112,0.914384,0.303745,-0.352079,0.670951,-0.577584, err_dist: 0.00972505, err_rot_angle: 0.00439332, num_iterations: 21, total_time: 121.248, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0853466,0.0321572,-0.0410106,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00792579,0.00176019,-0.00375077,0.999997,-0.0011419,0.00195872,-2.58465e-05, solved_pose: -0.82281,1.30124,0.912387,0.303832,-0.351874,0.670766,-0.577878, err_dist: 0.00894342, err_rot_angle: 0.00453484, num_iterations: 30, total_time: 189.548, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0373647,0.00757588,0.0924472,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00729773,0.00334981,-0.00241475,0.999998,-0.000825961,0.00177443,6.94374e-05, solved_pose: -0.82335,1.30237,0.914157,0.303542,-0.351927,0.670972,-0.57776, err_dist: 0.00838505, err_rot_angle: 0.00391696, num_iterations: 33, total_time: 249.07, convergence: 1`

* changes: downscale by 4, 0.1 offset

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.05177,0.00665848,0.0852967,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.0151129,0.0112302,0.00119651,0.999992,0.00112247,0.00394955,-9.65256e-05, solved_pose: -0.829687,1.29723,0.922463,0.304414,-0.353659,0.671494,-0.575633, err_dist: 0.0188666, err_rot_angle: 0.0082142, num_iterations: 25, total_time: 36.5242, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0917203,0.0365421,-0.0158764,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00791245,0.00222469,-0.00511611,0.999996,-0.000798326,0.00271508,0.000154501, solved_pose: -0.821619,1.30045,0.912643,0.304115,-0.352535,0.670672,-0.577435, err_dist: 0.00968146, err_rot_angle: 0.00566846, num_iterations: 28, total_time: 38.9947, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0184381,0.0134534,-0.0973604,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00194534,0.000408362,-0.00455103,0.999997,-0.00162375,0.00124719,0.00103398, solved_pose: -0.818931,1.30583,0.910947,0.302911,-0.352029,0.670331,-0.578772, err_dist: 0.00496618, err_rot_angle: 0.00458745, num_iterations: 27, total_time: 42.233, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0977698,0.0208044,0.00287079,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00980742,0.00570952,-0.00169465,0.999996,-0.00010308,0.00297917,6.03301e-06, solved_pose: -0.825047,1.3005,0.916587,0.304134,-0.352798,0.671046,-0.57683, err_dist: 0.0114741, err_rot_angle: 0.00596194, num_iterations: 27, total_time: 36.5268, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0988953,0.00247657,0.0146148,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.0125683,0.00810368,-0.000108265,0.999994,0.00043068,0.00347261,-3.26327e-05, solved_pose: -0.827593,1.29889,0.919184,0.3043,-0.353218,0.671218,-0.576286, err_dist: 0.0149547, err_rot_angle: 0.00699875, num_iterations: 20, total_time: 27.4532, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0597197,0.0360927,-0.0716301,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.0058216,-0.00561501,-0.0109667,0.999994,-0.0026596,0.00237191,0.000286257, solved_pose: -0.816606,1.29953,0.904064,0.304461,-0.351863,0.669653,-0.578845, err_dist: 0.0136267, err_rot_angle: 0.00715023, num_iterations: 17, total_time: 20.3051, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0910014,0.0411416,0.00511068,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00869465,0.00311287,-0.00393211,0.999996,-0.000656798,0.00281411,1.63765e-05, solved_pose: -0.822917,1.30039,0.913692,0.304212,-0.352543,0.670772,-0.577264, err_dist: 0.0100373, err_rot_angle: 0.00577959, num_iterations: 20, total_time: 29.3722, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0644758,0.0124351,0.0754204,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.0143604,0.0132345,0.00200769,0.999992,0.00165335,0.0037552,-1.05631e-05, solved_pose: -0.829665,1.29813,0.924572,0.304047,-0.353765,0.67183,-0.57537, err_dist: 0.0196317, err_rot_angle: 0.00820618, num_iterations: 15, total_time: 22.7869, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0789993,0.0531323,0.0305952,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.00608729,-0.002427,-0.00840194,0.999995,-0.001952,0.00230426,0.000417388, solved_pose: -0.818474,1.30049,0.907579,0.304092,-0.352126,0.670037,-0.578434, err_dist: 0.0106554, err_rot_angle: 0.00609726, num_iterations: 28, total_time: 59.4955, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.00291504,0.00142941,0.0999473,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 20, err: -0.015856,0.0127423,0.000642846,0.99999,0.00150169,0.00411135,-0.000110275, solved_pose: -0.829454,1.29617,0.92387,0.304397,-0.353857,0.671669,-0.575316, err_dist: 0.0203517, err_rot_angle: 0.00875684, num_iterations: 22, total_time: 28.4699, convergence: 1`

* no changes, 0.15 offset

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.00984484,0.00634031,-0.149542,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00332572,0.0014009,-0.00242856,0.999998,-0.00126223,0.00120558,0.00016496, solved_pose: -0.821324,1.3058,0.912234,0.303258,-0.351531,0.670858,-0.578282, err_dist: 0.00434981, err_rot_angle: 0.00350649, num_iterations: 24, total_time: 623.557, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0476187,0.00250605,0.142219,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00810764,0.00170825,-0.00351555,0.999997,-0.00109254,0.00219614,-0.000109269, solved_pose: -0.823107,1.30123,0.912368,0.304022,-0.35197,0.670752,-0.577736, err_dist: 0.00900062, err_rot_angle: 0.00491065, num_iterations: 25, total_time: 755.736, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0281088,0.0014574,-0.147336,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.000941601,0.00384397,2.66815e-05,0.999999,-0.000793735,0.000651524,0.000358084, solved_pose: -0.821667,1.30894,0.915013,0.30261,-0.351483,0.671229,-0.57822, err_dist: 0.0039577, err_rot_angle: 0.00217506, num_iterations: 17, total_time: 427.709, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.131305,0.0511378,0.0514202,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00774284,0.0008451,-0.00415649,0.999997,-0.00128739,0.00214243,-0.000106326, solved_pose: -0.822483,1.30124,0.911422,0.304053,-0.351882,0.670655,-0.577887, err_dist: 0.00882849, err_rot_angle: 0.00500348, num_iterations: 46, total_time: 1519.94, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.13297,0.021306,0.0660686,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00744986,0.00428599,-0.00116134,0.999998,-0.000563889,0.00198467,-7.19921e-06, solved_pose: -0.824338,1.30287,0.915266,0.303636,-0.352076,0.671086,-0.577486, err_dist: 0.00867288, err_rot_angle: 0.00412647, num_iterations: 29, total_time: 1333.47, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.137675,0.0227945,-0.0550103,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00734491,0.00434955,-0.00107535,0.999998,-0.000553337,0.00195905,2.42775e-06, solved_pose: -0.824341,1.303,0.915342,0.303609,-0.352071,0.671097,-0.577491, err_dist: 0.00860364, err_rot_angle: 0.00407139, num_iterations: 31, total_time: 1622.49, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.114833,0.0773429,-0.0577181,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00777563,0.000868783,-0.00404116,0.999997,-0.00128817,0.00214447,-0.000116631, solved_pose: -0.822593,1.30128,0.911463,0.304061,-0.351876,0.670657,-0.577883, err_dist: 0.00880603, err_rot_angle: 0.00500869, num_iterations: 53, total_time: 2513.34, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.143646,0.0114214,-0.0416583,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00696535,0.00631409,0.000576696,0.999998,-0.000144334,0.00182998,7.09473e-05, solved_pose: -0.825228,1.30407,0.917532,0.30334,-0.352166,0.671348,-0.577283, err_dist: 0.00941894, err_rot_angle: 0.00367407, num_iterations: 35, total_time: 1793.98, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0391032,0.0187734,0.143591,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00773745,0.000569901,-0.00439967,0.999997,-0.00134459,0.00215042,-0.000112399, solved_pose: -0.822315,1.30113,0.911115,0.304082,-0.351865,0.670621,-0.57792, err_dist: 0.00891908, err_rot_angle: 0.00507735, num_iterations: 44, total_time: 2132.36, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.134291,0.0668284,0.000120592,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00774423,0.000564767,-0.00441193,0.999997,-0.00134593,0.00215175,-0.000112138, solved_pose: -0.82231,1.30112,0.911108,0.304083,-0.351866,0.67062,-0.577921, err_dist: 0.00893068, err_rot_angle: 0.00508101, num_iterations: 50, total_time: 2322.64, convergence: 1`

* changes: downscale by 2, 0.15 offset

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.124354,0.0179113,-0.0819465,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00795174,0.0013827,-0.00406823,0.999997,-0.0012102,0.00198302,-4.27345e-05, solved_pose: -0.822611,1.30107,0.911968,0.303882,-0.351856,0.670726,-0.57791, err_dist: 0.00903839, err_rot_angle: 0.00464706, num_iterations: 27, total_time: 201.548, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0259158,0.0107824,0.14735,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00824088,-0.00028084,-0.00544473,0.999997,-0.00154574,0.00210864,-0.000127924, solved_pose: -0.821853,1.3002,0.910121,0.304133,-0.35177,0.670523,-0.578066, err_dist: 0.0098811, err_rot_angle: 0.00523528, num_iterations: 27, total_time: 201.678, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.106683,0.0622857,0.0850836,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00824916,-0.000279102,-0.00541734,0.999997,-0.00155057,0.00210641,-0.000130407, solved_pose: -0.821879,1.30021,0.910127,0.304135,-0.351765,0.670522,-0.578069, err_dist: 0.0098729, err_rot_angle: 0.00523765, num_iterations: 48, total_time: 343.768, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.148623,0.012908,0.0156408,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00811123,0.0016012,-0.00391251,0.999997,-0.00117395,0.00199962,-3.18993e-05, solved_pose: -0.822802,1.30101,0.912206,0.303874,-0.351884,0.670738,-0.577883, err_dist: 0.00914678, err_rot_angle: 0.00463796, num_iterations: 31, total_time: 208.04, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.135657,0.0352337,0.0534402,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.0082657,-0.000137272,-0.00533,0.999997,-0.00151661,0.00210966,-0.000121179, solved_pose: -0.821943,1.30023,0.91028,0.30412,-0.351784,0.670537,-0.578048, err_dist: 0.00983613, err_rot_angle: 0.00520211, num_iterations: 28, total_time: 217.362, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0763322,0.010065,-0.128733,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00753405,0.000362519,-0.00437151,0.999997,-0.00142288,0.00183512,5.17577e-06, solved_pose: -0.822253,1.30133,0.910917,0.30383,-0.351738,0.67063,-0.57812, err_dist: 0.008718, err_rot_angle: 0.00464427, num_iterations: 17, total_time: 77.4541, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.131907,0.0586855,-0.0406999,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00825182,-7.88515e-06,-0.00524344,0.999997,-0.00149098,0.0021012,-0.000112644, solved_pose: -0.82199,1.30028,0.91042,0.3041,-0.351792,0.670552,-0.578036, err_dist: 0.00977682, err_rot_angle: 0.00515782, num_iterations: 33, total_time: 253.522, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0736045,0.0147439,0.129865,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00821874,-0.000260031,-0.00540277,0.999997,-0.00154298,0.00210169,-0.000126447, solved_pose: -0.821872,1.30024,0.910148,0.304127,-0.351768,0.670526,-0.578067, err_dist: 0.00983897, err_rot_angle: 0.00522069, num_iterations: 35, total_time: 239.992, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.122886,0.0665549,-0.0544926,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00824297,-0.000338968,-0.00547884,0.999997,-0.00155753,0.00211004,-0.000131263, solved_pose: -0.821833,1.30019,0.910058,0.30414,-0.351765,0.670517,-0.578072, err_dist: 0.0099035, err_rot_angle: 0.00525182, num_iterations: 32, total_time: 248.375, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.149691,0.00862257,0.00425897,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00802798,0.00198075,-0.0036375,0.999997,-0.0010892,0.00197773,-1.61318e-05, solved_pose: -0.822933,1.3012,0.912622,0.303821,-0.351907,0.670788,-0.577839, err_dist: 0.00903346, err_rot_angle: 0.00451576, num_iterations: 24, total_time: 166.304, convergence: 1`

* changes: downscale by 4, 0.15 offset

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.113156,0.0280178,0.0943964,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00957554,0.00447869,-0.00326604,0.999996,-0.000273249,0.00293076,7.04164e-05, solved_pose: -0.823787,1.29992,0.915138,0.304124,-0.352762,0.67094,-0.576981, err_dist: 0.0110642, err_rot_angle: 0.00588863, num_iterations: 19, total_time: 30.5005, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0348166,0.0252853,-0.143696,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00475641,0.000304218,-0.0052296,0.999997,-0.00146206,0.00191053,0.000603923, solved_pose: -0.819982,1.30314,0.910742,0.303548,-0.352172,0.670375,-0.5783, err_dist: 0.00707564, err_rot_angle: 0.00496084, num_iterations: 30, total_time: 47.9254, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.138792,0.0549375,-0.0147902,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00732749,-0.00616152,-0.0115426,0.999993,-0.00270518,0.00265979,0.00012849, solved_pose: -0.817046,1.29802,0.903437,0.304762,-0.351909,0.669594,-0.578726, err_dist: 0.0149963, err_rot_angle: 0.00759186, num_iterations: 29, total_time: 47.6385, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.146374,0.0180505,0.0273655,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.0166745,0.0165776,0.002941,0.999989,0.00234824,0.00417346,5.61921e-05, solved_pose: -0.831283,1.29643,0.928016,0.304045,-0.35426,0.672079,-0.574775, err_dist: 0.0236961, err_rot_angle: 0.00957818, num_iterations: 17, total_time: 24.4962, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.129838,0.0387704,0.0643347,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00793875,0.00382905,-0.00298472,0.999996,-0.00052726,0.00263451,0.000110841, solved_pose: -0.823176,1.30148,0.914543,0.303991,-0.352542,0.670869,-0.577268, err_dist: 0.00930559, err_rot_angle: 0.00537809, num_iterations: 36, total_time: 48.6801, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.102652,0.0505596,0.0969857,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00661122,-0.00147595,-0.00718702,0.999995,-0.00177336,0.00240551,0.000341993, solved_pose: -0.819646,1.30067,0.908694,0.304141,-0.352188,0.670136,-0.578256, err_dist: 0.00987623, err_rot_angle: 0.00601607, num_iterations: 42, total_time: 70.2068, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.137173,0.0120209,-0.0594903,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.0113384,0.00798423,-0.0007812,0.999994,0.000507497,0.00331211,3.57633e-06, solved_pose: -0.826365,1.29954,0.918968,0.304145,-0.353173,0.671298,-0.576302, err_dist: 0.0138894, err_rot_angle: 0.00670155, num_iterations: 21, total_time: 25.0488, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.0955679,0.0488837,0.104772,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00742776,-0.0104883,-0.0152858,0.99999,-0.00364709,0.002761,0.000134989, solved_pose: -0.814541,1.29618,0.898641,0.305156,-0.351686,0.669015,-0.579324, err_dist: 0.0199708, err_rot_angle: 0.00915265, num_iterations: 33, total_time: 50.4427, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.147619,0.0253632,-0.00808524,1,0,0,-0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00871352,0.00319455,-0.00369648,0.999996,-0.000650001,0.00282258,5.62715e-05, solved_pose: -0.823109,1.3005,0.913809,0.304192,-0.352576,0.67076,-0.577268, err_dist: 0.00998972, err_rot_angle: 0.00579401, num_iterations: 31, total_time: 42.7464, convergence: 1`

`cam_index: 0, ref_img_index: 1, blurred_img_index: 2, sigma: 0,initial_offset: 0.138542,0.0486388,0.0306645,1,0,0,0, initial_offset_dist: 0.15, initial_offset_rot_angle: 0, n_images: 20, err: -0.00749608,-0.00316906,-0.00853151,0.999994,-0.00210821,0.0026432,0.000157313, solved_pose: -0.819247,1.29932,0.906824,0.304525,-0.3521,0.669935,-0.578341, err_dist: 0.0117907, err_rot_angle: 0.0067693, num_iterations: 43, total_time: 68.388, convergence: 1`


* no changes, 0.20 offset

* no changes, 0.25 offset

