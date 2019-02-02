### Docker

The [docker](https://gitlab.com/jdiep/semester-thesis/tree/3-neural-mesh-reprojection/docker) folder contains a Dockerfile able to run the [camera tracker](https://gitlab.com/jdiep/semester-thesis/tree/3-neural-mesh-reprojection/pose-estimator) from Aebi, Milano and Schnetzler. The following commands can be used: 


***Accessing root***
```bash
sudo -s
```

***Stop all containers***
```bash
docker stop $(docker ps -a -q)
```

***Delete all containers***
```bash
docker rm $(docker ps -a -q)
```

***Delete all images***
```bash
docker rmi $(docker images -q)
```

***Building image***
```bash
docker build -t st-ubuntu .
```

***Run image***
```bash
docker run -it st-ubuntu
```

***Show all images***
```bash
docker images
```

***Show all containers***
```bash
docker ps -a
```

***Copy file to host machine***
```bash
docker cp container_name:/home/semester-thesis/file /home/johann
```

***After running bash***
```bash
apt-get update
apt-get install libmrpt-dev
git clone https://gitlab.com/jdiep/semester-thesis -b 2-dockerfile-to-install-all-dependencies-for-the-project
```

***Run multiple terminal of same container***
```bash
docker exec -it container_name bash
```

## Authors

* Johann Diep (MSc Student Mechanical Engineering, ETH Zurich, jdiep@student.ethz.ch)