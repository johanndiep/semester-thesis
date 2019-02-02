## Docker

The [docker](https://gitlab.com/jdiep/semester-thesis/tree/3-neural-mesh-reprojection/docker) folder contains a Dockerfile able to run the [motion blur aware camera pose tracker](https://gitlab.com/jdiep/semester-thesis/tree/3-neural-mesh-reprojection/pose-estimator) developed by Aebi, Milano and Schnetzler in the context of the lecture 3D Vision at ETH Zurich in spring 2018. The following commands can be used to start the Docker container: 


#### Accessing root:
```bash
sudo -s
```

#### Stopping all containers:
```bash
docker stop $(docker ps -a -q)
```

#### Deleting all containers:
```bash
docker rm $(docker ps -a -q)
```

#### Deleting all images:
```bash
docker rmi $(docker images -q)
```

#### Building image:
```bash
docker build -t st-ubuntu .
```

#### Run image:
```bash
docker run -it st-ubuntu
```

#### Displaying all images:
```bash
docker images
```

#### Displaying all containers:
```bash
docker ps -a
```

#### Copying file to host machine:
```bash
docker cp container_name:/home/semester-thesis/<file> /home/<host>
```

#### After running bash (intermediate solution):
```bash
apt-get update
apt-get install libmrpt-dev
git clone https://gitlab.com/jdiep/semester-thesis -b 2-dockerfile-to-install-all-dependencies-for-the-project
```

#### Running multiple terminal of same container:
```bash
docker exec -it container_name bash
```

## Authors

* Johann Diep (MSc Student Mechanical Engineering, ETH Zurich, jdiep@student.ethz.ch)