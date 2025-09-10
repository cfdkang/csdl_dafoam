# csdl_dafoam: CSDL - DAFoam interface and 2-D airfoil aerodynamic shape optimization tutorial script

**csdl_dafoam** is a CSDL-DAFoam interface to implement aerodynamic shape optimization. This interface utilizes **lsdo_geo** for geometric parametrization, **IDWarp** for mesh deformation, **modopt** for optimizer, **CSDL** for automatic differentiation, and **DAFoam** for primal and adjoint computations. 

## Instructions to run the script for users
To install **csdl_fwh**, first clone the repository and install using pip. On the terminal or command line, run

(1) pull the docker images from Docker Hub
```bash
docker pull cfdkang/csdl_dafoam
```

(2) Check the pulled image
```bash
docker images
```

(3) run (mount) docker image
```bash
docker run -it --rm -u dafoamuser --mount \
"type=bind,src=$(pwd),target=/home/dafoamuser/mount" \
-w /home/dafoamuser/mount dafoam/cfdkang/csdl_dafoam:latest bash
```

(4) You are ready to run the tutorial script
```bash
./run.sh
```

## Instructions to run the script for users
To install **csdl_fwh**, first clone the repository and install using pip. On the terminal or command line, run

(1) Download the dockerfile:
```bash
https://github.com/cfdkang/Dockerfile
```

(2) Build the dockerfile
```bash
DOCKER_BUILDKIT=1 docker build --progress=plain -t csdl_dafoam:latest .
```

(3) Check the pulled image
```bash
docker images
```

(4) run (mount) docker image
```bash
docker run -it --rm -u dafoamuser --mount \
"type=bind,src=$(pwd),target=/home/dafoamuser/mount" \
-w /home/dafoamuser/mount dafoam/csdl_dafoam:latest bash
```

(5) You are ready to run the tutorial script
```bash
./run.sh
```