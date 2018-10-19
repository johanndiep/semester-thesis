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

* Parameters: cam_index 0, dataset_path /home/semester-thesis/RelisticRendering-dataset/, ref_img_index 1, blurred_img_index 2, n_images 20, initial_offset_rot 0, output_file "results", sigma 0

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

* Sequential testing: cam_index = 0, ref_img_index = 1, blurred_img_index = 3, sigma = 0, initial_offset_norm = 0.1, initial_offset_rot = 0, n_images = 5, converged = true

`cam_index: 0, ref_img_index: 1, blurred_img_index: 3, sigma: 0,initial_offset: 0.0219132,0.0102488,0.0970298,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 5, err: -0.0197611,0.00846976,-0.00269012,0.999991,-0.000530154,0.00419576,-0.000714922, solved_pose: -0.939864,1.00142,0.92068,0.324736,-0.385831,0.661285,-0.555322, err_dist: 0.0216674, err_rot_angle: 0.00857827, num_iterations: 31, total_time: 228.234, convergence: 1` (no layers)

`cam_index: 0, ref_img_index: 1, blurred_img_index: 3, sigma: 0,initial_offset: 0,0,-0,1,0,0,-0, initial_offset_dist: 0, initial_offset_rot_angle: 0, n_images: 5, err: -0.00614251,0.00998383,0.00353481,0.999999,-0.000302075,0.00138421,-0.00039391, solved_pose: -0.938299,1.0162,0.923268,0.322609,-0.384555,0.662198,-0.556359, err_dist: 0.0122435, err_rot_angle: 0.00294106, num_iterations: 11, total_time: 42.3846, convergence: 1` (ideal)

`Layer: 3
cam_index: 0, ref_img_index: 1, blurred_img_index: 3, sigma: 0,initial_offset: 0.0503358,0.0321539,0.0802024,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 5, err: -0.0205652,0.00590534,-0.00681956,0.99999,-0.00122256,0.00438759,-0.000426529, solved_pose: -0.937094,0.998947,0.917463,0.324968,-0.385906,0.660726,-0.555799, err_dist: 0.0224568, err_rot_angle: 0.00914936, num_iterations: 38, total_time: 31.9818, convergence: 1, image_scale: 3
=============================================================
Layer: 2
initial_pose: -0.937094,0.998947,0.917463,0.324968,-0.385906,0.660726,-0.555799, err: -0.0228049,0.00174173,-0.00859384,0.999985,-0.00198634,0.00499859,-0.000830503, solved_pose: -0.937265,0.996485,0.913047,0.32589,-0.385732,0.660257,-0.555938, err_dist: 0.0244326, err_rot_angle: 0.0108851, num_iterations: 33, total_time: 56.9889, convergence: 1, image_scale: 2
=============================================================
Layer: 1
initial_pose: -0.937265,0.996485,0.913047,0.32589,-0.385732,0.660257,-0.555938, err: -0.0196642,0.0077838,-0.00299287,0.999991,-0.00068515,0.00421196,-0.000772833, solved_pose: -0.939662,1.00142,0.919952,0.324839,-0.385752,0.661216,-0.5554, err_dist: 0.0213594, err_rot_angle: 0.0086735, num_iterations: 43, total_time: 327.005, convergence: 1, image_scale: 1`

`Layer: 5
cam_index: 0, ref_img_index: 1, blurred_img_index: 3, sigma: 0,initial_offset: 0.0794841,0.0240947,-0.0556931,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 5, err: -0.0214251,-0.0033724,-0.0098222,0.999985,-0.00340335,0.00428928,-0.000727401, solved_pose: -0.936305,0.997521,0.907836,0.325907,-0.38495,0.659656,-0.557183, err_dist: 0.0238093, err_rot_angle: 0.0110472, num_iterations: 13, total_time: 2.65959, convergence: 1, image_scale: 5
=============================================================
Layer: 4
initial_pose: -0.936305,0.997521,0.907836,0.325907,-0.38495,0.659656,-0.557183, err: -0.0209143,-0.00026496,-0.00265765,0.999988,-0.00265511,0.00401718,8.04053e-05, solved_pose: -0.941765,1.00118,0.912126,0.32499,-0.385575,0.659852,-0.557053, err_dist: 0.0210842, err_rot_angle: 0.00963202, num_iterations: 16, total_time: 5.0119, convergence: 1, image_scale: 4
=============================================================
Layer: 3
initial_pose: -0.941765,1.00118,0.912126,0.32499,-0.385575,0.659852,-0.557053, err: -0.0203893,0.00295106,-0.00787058,0.999989,-0.00188299,0.00425674,-0.000221929, solved_pose: -0.93655,0.998844,0.914393,0.325021,-0.385757,0.660322,-0.556352, err_dist: 0.022054, err_rot_angle: 0.00931985, num_iterations: 26, total_time: 16.1588, convergence: 1, image_scale: 3
=============================================================
Layer: 2
initial_pose: -0.93655,0.998844,0.914393,0.325021,-0.385757,0.660322,-0.556352, err: -0.0213965,0.00495984,-0.00572331,0.999989,-0.00143679,0.00445257,-0.000347481, solved_pose: -0.938582,0.998834,0.916724,0.325049,-0.385926,0.660556,-0.555941, err_dist: 0.0226973, err_rot_angle: 0.0093831, num_iterations: 23, total_time: 29.9667, convergence: 1, image_scale: 2
=============================================================
Layer: 1
initial_pose: -0.938582,0.998834,0.916724,0.325049,-0.385926,0.660556,-0.555941, err: -0.0200095,0.00776311,-0.00309608,0.999991,-0.000701051,0.00421831,-0.000653878, solved_pose: -0.939744,1.00107,0.919918,0.324783,-0.385829,0.661159,-0.555446, err_dist: 0.0216849, err_rot_angle: 0.00865177, num_iterations: 23, total_time: 154.14, convergence: 1, image_scale: 1`

`Layer: 10
cam_index: 0, ref_img_index: 1, blurred_img_index: 3, sigma: 0,initial_offset: 0.0295079,0.00647505,-0.0953276,1,0,0,0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 5, err: 0.0117963,-0.00245607,0.0401179,0.99999,-0.0016308,-0.00375433,0.00157087, solved_pose: -0.962267,1.05102,0.917151,0.318618,-0.382565,0.662349,-0.55984, err_dist: 0.0418884, err_rot_angle: 0.00876864, num_iterations: 10, total_time: 0.424607, convergence: 1, image_scale: 10
=============================================================
Layer: 9
initial_pose: -0.962267,1.05102,0.917151,0.318618,-0.382565,0.662349,-0.55984, err: 0.0131883,0.00822943,0.0192014,0.999988,-0.00132087,-0.00338597,0.00329885, solved_pose: -0.942176,1.04096,0.92408,0.31778,-0.384014,0.661738,-0.560047, err_dist: 0.0247053, err_rot_angle: 0.00981674, num_iterations: 23, total_time: 1.35206, convergence: 1, image_scale: 9
=============================================================
Layer: 8
initial_pose: -0.942176,1.04096,0.92408,0.31778,-0.384014,0.661738,-0.560047, err: 0.0100995,0.0169742,0.0166985,0.999994,0.0011156,-0.00313432,0.000680602, solved_pose: -0.940232,1.03624,0.932418,0.318471,-0.383205,0.664023,-0.557498, err_dist: 0.0258643, err_rot_angle: 0.0067917, num_iterations: 29, total_time: 2.3374, convergence: 1, image_scale: 8
=============================================================
Layer: 7
initial_pose: -0.940232,1.03624,0.932418,0.318471,-0.383205,0.664023,-0.557498, err: 0.000219627,0.0100466,0.00300864,0.999999,-0.000325356,0.00081578,-0.000556089, solved_pose: -0.934684,1.02147,0.923235,0.322332,-0.384124,0.66243,-0.55654, err_dist: 0.0104898, err_rot_angle: 0.00207903, num_iterations: 25, total_time: 2.49321, convergence: 1, image_scale: 7
=============================================================
Layer: 6
initial_pose: -0.934684,1.02147,0.923235,0.322332,-0.384124,0.66243,-0.55654, err: 0.00219293,0.00873936,0.0114911,0.999997,-0.0016564,-0.000547209,0.00177658, solved_pose: -0.941133,1.02752,0.923374,0.32064,-0.384482,0.661229,-0.558694, err_dist: 0.0146025, err_rot_angle: 0.0049797, num_iterations: 14, total_time: 1.91299, convergence: 1, image_scale: 6
=============================================================
Layer: 5
initial_pose: -0.941133,1.02752,0.923374,0.32064,-0.384482,0.661229,-0.558694, err: 0.00935758,0.00589331,0.0100975,0.999996,-0.00178186,-0.0022928,0.000491624, solved_pose: -0.936758,1.03327,0.920309,0.320247,-0.382618,0.662213,-0.559034, err_dist: 0.0149751, err_rot_angle: 0.00589022, num_iterations: 13, total_time: 2.72159, convergence: 1, image_scale: 5
=============================================================
Layer: 4
initial_pose: -0.936758,1.03327,0.920309,0.320247,-0.382618,0.662213,-0.559034, err: 0.00556349,0.00117718,0.0085324,0.999996,-0.0024464,-0.00128098,0.000325122, solved_pose: -0.938052,1.02959,0.915402,0.321266,-0.382857,0.661582,-0.559033, err_dist: 0.0102538, err_rot_angle: 0.00556112, num_iterations: 32, total_time: 11.5757, convergence: 1, image_scale: 4
=============================================================
Layer: 3
initial_pose: -0.938052,1.02959,0.915402,0.321266,-0.382857,0.661582,-0.559033, err: 0.00213691,0.00898749,0.00312625,1,-0.000801042,-0.000132338,0.000335952, solved_pose: -0.933984,1.0233,0.922205,0.32139,-0.384034,0.662127,-0.557507, err_dist: 0.00975268, err_rot_angle: 0.00175732, num_iterations: 24, total_time: 14.7869, convergence: 1, image_scale: 3
=============================================================
Layer: 2
initial_pose: -0.933984,1.0233,0.922205,0.32139,-0.384034,0.662127,-0.557507, err: 0.000976495,0.00627709,0.00353576,0.999999,-0.00124313,0.00017392,-1.8478e-05, solved_pose: -0.935328,1.02272,0.919607,0.32196,-0.383828,0.661919,-0.557568, err_dist: 0.00727028, err_rot_angle: 0.00251076, num_iterations: 13, total_time: 16.7649, convergence: 1, image_scale: 2
=============================================================
Layer: 1
initial_pose: -0.935328,1.02272,0.919607,0.32196,-0.383828,0.661919,-0.557568, err: 0.000961524,0.00634727,0.00377453,0.999999,-0.00112352,4.45199e-05,-0.000301058, solved_pose: -0.935529,1.02281,0.919718,0.321985,-0.383607,0.662136,-0.557448, err_dist: 0.00744711, err_rot_angle: 0.00232801, num_iterations: 13, total_time: 67.2802, convergence: 1, image_scale: 1`

`Layer: 10
cam_index: 0, ref_img_index: 1, blurred_img_index: 3, sigma: 0,initial_offset: 0.068889,0.0341533,0.0639363,1,0,0,-0, initial_offset_dist: 0.1, initial_offset_rot_angle: 0, n_images: 5, err: -0.0421135,-0.0232205,-0.051796,0.999934,-0.00751075,0.00868639,-0.000107053, solved_pose: -0.912526,0.960884,0.88163,0.330037,-0.38647,0.655684,-0.558386, err_dist: 0.0706793, err_rot_angle: 0.022968, num_iterations: 11, total_time: 0.550468, convergence: 1, image_scale: 10
=============================================================
Layer: 9
initial_pose: -0.912526,0.960884,0.88163,0.330037,-0.38647,0.655684,-0.558386, err: -0.0521627,-0.0240713,-0.0468141,0.999912,-0.00768529,0.0108471,-0.000379342, solved_pose: -0.921504,0.95461,0.881518,0.33168,-0.387428,0.654982,-0.557572, err_dist: 0.0741075, err_rot_angle: 0.026599, num_iterations: 15, total_time: 0.709379, convergence: 1, image_scale: 9
=============================================================
Layer: 8
initial_pose: -0.921504,0.95461,0.881518,0.33168,-0.387428,0.654982,-0.557572, err: -0.0494002,-0.0230882,-0.037153,0.999916,-0.00619348,0.0110539,-0.00285781, solved_pose: -0.928445,0.961696,0.883681,0.332625,-0.386381,0.656701,-0.55571, err_dist: 0.0659833, err_rot_angle: 0.0259787, num_iterations: 12, total_time: 0.882567, convergence: 1, image_scale: 8
=============================================================
Layer: 7
initial_pose: -0.928445,0.961696,0.883681,0.332625,-0.386381,0.656701,-0.55571, err: -0.0461815,-0.011792,-0.0365954,0.999918,-0.00511416,0.011619,-0.00195466, solved_pose: -0.925792,0.963841,0.894941,0.332083,-0.387642,0.656775,-0.555069, err_dist: 0.0600916, err_rot_angle: 0.0256893, num_iterations: 23, total_time: 2.47369, convergence: 1, image_scale: 7
=============================================================
Layer: 6
initial_pose: -0.925792,0.963841,0.894941,0.332083,-0.387642,0.656775,-0.555069, err: -0.0365492,-0.0188532,-0.0371077,0.99993,-0.00665323,0.00968354,-0.00112163, solved_pose: -0.921859,0.972514,0.888135,0.330932,-0.386627,0.656228,-0.557107, err_dist: 0.055392, err_rot_angle: 0.0236052, num_iterations: 26, total_time: 3.76885, convergence: 1, image_scale: 6
=============================================================
Layer: 5
initial_pose: -0.921859,0.972514,0.888135,0.330932,-0.386627,0.656228,-0.557107, err: -0.0277783,-0.0201857,-0.0275866,0.999952,-0.00644738,0.00698227,-0.00244477, solved_pose: -0.926249,0.984829,0.888319,0.329806,-0.38432,0.657734,-0.557595, err_dist: 0.0440468, err_rot_angle: 0.0196266, num_iterations: 21, total_time: 6.21072, convergence: 1, image_scale: 5
=============================================================
Layer: 4
initial_pose: -0.926249,0.984829,0.888319,0.329806,-0.38432,0.657734,-0.557595, err: -0.0205666,0.00718473,-0.00167213,0.999991,-0.00113317,0.00405942,6.09701e-05, solved_pose: -0.941328,1.00134,0.919625,0.324445,-0.386076,0.660695,-0.556023, err_dist: 0.0218495, err_rot_angle: 0.00843013, num_iterations: 39, total_time: 20.664, convergence: 1, image_scale: 4
=============================================================
Layer: 3
initial_pose: -0.941328,1.00134,0.919625,0.324445,-0.386076,0.660695,-0.556023, err: -0.0205194,0.00710386,-0.00620656,0.99999,-0.000965814,0.00434197,-0.000385846, solved_pose: -0.937418,0.999182,0.918747,0.324817,-0.385991,0.660869,-0.55566, err_dist: 0.0225839, err_rot_angle: 0.00892962, num_iterations: 32, total_time: 29.6391, convergence: 1, image_scale: 3
=============================================================
Layer: 2
initial_pose: -0.937418,0.999182,0.918747,0.324817,-0.385991,0.660869,-0.55566, err: -0.0225529,0.00427451,-0.00707636,0.999987,-0.00147749,0.00482642,-0.000669728, solved_pose: -0.938076,0.997225,0.915801,0.325492,-0.385907,0.660535,-0.55572, err_dist: 0.0240204, err_rot_angle: 0.0101835, num_iterations: 31, total_time: 57.7476, convergence: 1, image_scale: 2
=============================================================
Layer: 1
initial_pose: -0.938076,0.997225,0.915801,0.325492,-0.385907,0.660535,-0.55572, err: -0.020418,0.00801035,-0.00326473,0.99999,-0.000613289,0.0043488,-0.000741242, solved_pose: -0.939758,1.00061,0.920127,0.324884,-0.385872,0.661199,-0.555309, err_dist: 0.0221747, err_rot_angle: 0.00890792, num_iterations: 33, total_time: 277.373, convergence: 1, image_scale: 1`
