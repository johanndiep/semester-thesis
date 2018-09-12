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

###building image

```bash
docker build -t st-ubuntu .
```

###run image

```bash
docker run -it st-ubuntu
```

Implementation
------------

* Coarse-to-fine image resolution: Implementation of the coarse-to-fine method to run the minimalization step more efficiently.