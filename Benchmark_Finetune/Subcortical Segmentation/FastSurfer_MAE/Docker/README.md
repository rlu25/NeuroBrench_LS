# FastSurfer Docker Support
## Pull FastSurfer from DockerHub

We provide pre-built Docker images with support for nVidia GPU-acceleration and for CPU-only use on [Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags). 
In order to quickly get the latest Docker image, simply execute:

```bash 
docker pull deepmi/fastsurfer
```

This will download the newest, official FastSurfer image with support for nVidia GPUs.

Image are named and tagged as follows: `deepmi/fastsurfer:<support>-<version>`, where `<support>` is `gpu` for support of nVidia GPUs and `cpu` without hardware acceleration (the latter is smaller and thus faster to download).
Similarly, `<version>` can be a version string (`latest` or `v#.#.#`, where `#` are digits, for example `v2.2.2`), for example:

```bash 
docker pull deepmi/fastsurfer:cpu-v2.2.2
```

### Running the official Docker Image
After pulling the image, you can start a FastSurfer container and process a T1-weighted image (both segmentation and surface reconstruction) with the following command:

```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/t1-weighted.nii.gz \
                      --sid subjectX --sd /output \
                      --threads 4 --3T # and more flags
```

#### Docker Flags
* `--gpus`: This flag is used to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use multiple specific ones (e.g. GPU 0, 1 and 3), set `--gpus "device=0,1,3"`.
* `-v`: This commands mount your data, output and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license).
* `--rm`: The flag takes care of removing the container once the analysis finished. 
* `-d`: This is optional. You can add this flag to run in detached mode (no screen output and you return to shell)
* `--user $(id -u):$(id -g)`: Run the container with your account (your user-id and group-id), which are determined by `$(id -u)` and `$(id -g)`, respectively. Running the docker container as root `-u 0:0` is strongly discouraged.

#### Advanced Docker Flags
* `--group-add <list of groups>`: If additional user groups are required to access files, additional groups may be added via `--group-add <group id>[,...]` or `--group-add $(id -G <group name>)`. 

#### FastSurfer Flags
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the `my_fs_license_dir` that was mapped above. 
* The `--t1` points to the t1-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_fastsurfer_analysis => /output)
* [more flags](../doc/overview/FLAGS.md#fastsurfer-flags)

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments. 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory (specified via `--sd`). So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

All other available flags are identical to the ones explained on the main page [README](../README.md).

### Docker Best Practice
* Do not mount the user home directory into the docker container as the home directory.
  
  Why? If the user inside the docker container has access to a user directory, settings from that directory might bleed into the FastSurfer pipeline. For example, before FastSurfer 2.2 python packages installed in the user directory would replace those installed inside the image potentially causing incompatibilities. Since FastSurfer 2.2, `docker run ... --version +pip` outputs the FastSurfer version including a full list of python packages. 

  How? Docker does not mount the home directory by default, so unless you manually set the `HOME` environment variable, all should be fine. 

## FastSurfer Docker Image Creation

Within this directory, we currently provide a build script and Dockerfile to create multiple Docker images for users (usually developers) who wish to create their own Docker images for 3 platforms:

* Nvidia / CUDA (Example 1)
* CPU (Example 2)
* AMD / rocm (experimental, Example 3)

To run only the surface pipeline or only the segmentation pipeline, the entrypoint to these images has to be adapted, which is possible through
-  for the segmentation pipeline: `--entrypoint "python /fastsurfer/FastSurferCNN/run_prediction.py"`
-  for the surface pipeline: `--entrypoint "/fastsurfer/recon_surf/recon-surf.sh"`

Note, for many HPC users with limited GPUs or with very large datasets, it may be most efficient to run the full pipeline on the CPU, trading a longer run-time for the segmentation with massive parallelization on the subject level. 

Also note, in order to run our Docker containers on a Mac, users need to increase docker memory to 10 GB by overwriting the settings under Docker Desktop --> Preferences --> Resources --> Advanced (slide the bar under Memory to 10 GB; see: [Change Docker Desktop settings on Mac](https://docs.docker.com/desktop/settings/mac/) for details). For the new Apple silicon chips (M1,etc), we noticed that a native install runs much faster than docker, because the Apple Accelerator (use the experimental MPS device via `--device mps`) can be used. There is no support for MPS-based acceleration through docker at the moment. 

### General build settings
The build script `build.py` supports additional args, targets and options, see `python Docker/build.py --help`.

Note, that the build script's main function is to select parameters for build args, but also create the FastSurfer-root/BUILD.info file, which will be used by FastSurfer to document the version (including git hash of the docker container). This BUILD.info file must exist for the docker build to be successful.
In general, if you specify `--dry_run` the command will not be executed but sent to stdout, so you can run `python build.py --device cuda --dry_run | bash` as well. Note, that build.py uses some dependencies from FastSurfer, so you will need to set the PYTHONPATH environment variable to the FastSurfer root (include of `FastSurferCNN` must be possible) and we only support Python 3.10.

By default, the build script will tag your image as `"fastsurfer:[{device}-]{version_tag}"`, where `{version_tag}` is `{version-identifer from pyproject.toml}_{current git-hash}` and `{device}` is the value to `--device` (omitted for `cuda`), but a custom tag can be specified by `--tag {tag_name}`. 

#### BuildKit
Note, we recommend using BuildKit to build docker images (e.g. `DOCKER_BUILDKIT=1` -- the build.py script already always adds this). To install BuildKit, run `wget -qO ~/.docker/cli-plugins/docker-buildx https://github.com/docker/buildx/releases/download/<version>/buildx-<version>.<platform>`, for example `wget -qO ~/.docker/cli-plugins/docker-buildx https://github.com/docker/buildx/releases/download/v0.12.1/buildx-v0.12.1.linux-amd64`. See also https://github.com/docker/buildx#manual-download.

### Example 1: Build GPU FastSurfer Image

In order to build your own Docker image for FastSurfer (FastSurferCNN + recon-surf; on GPU; including FreeSurfer) yourself simply execute the following command after traversing into the *Docker* directory: 

```bash
PYTHONPATH=<FastSurferRoot>
python build.py --device cuda --tag my_fastsurfer:cuda
```

The build script allows more specific options, that specify different CUDA options as well (see `build.py --help`).

For running the analysis, the command is the same as above for the prebuild option:
```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) my_fastsurfer:cuda \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/t1-weighted.nii.gz \
                      --sid subjectX --sd /output
```


### Example 2: Build CPU FastSurfer Image

In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on CPU; including FreeSurfer) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
PYTHONPATH=<FastSurferRoot>
python build.py --device cpu --tag my_fastsurfer:cpu
```

As you can see, only the `--device` to the build command is changed from `cuda` to `cpu`. 

For running the analysis, the command is basically the same as above, except for removing the `--gpus all` GPU option:
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) my_fastsurfer:cpu \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/t1-weighed.nii.gz \
           --sid subjectX --sd /output
```

FastSurfer will automatically detect, that no GPU is available and use the CPU.

### Example 3: Experimental Build for AMD GPUs

Here we build an experimental image to test performance when running on AMD GPUs. Note that you need a supported OS and Kernel version and supported GPU for the RocM to work correctly. You need to install the Kernel drivers into 
your host machine kernel (`amdgpu-install --usecase=dkms`) for the amd docker to work. For this follow:
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html#rocm-install-quick, https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/amdgpu-install.html#amdgpu-install-dkms and https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

```bash
PYTHONPATH=<FastSurferRoot>
python build.py --device rocm --tag my_fastsurfer:rocm
```

and run segmentation only:

```bash
docker run --rm --security-opt seccomp=unconfined \
           --device=/dev/kfd --device=/dev/dri --group-add video \
	   -v /home/user/my_mri_data:/data \
	   -v /home/user/my_fastsurfer_analysis:/output \
	   my_fastsurfer:rocm \
	   --t1 /data/subjectX/t1-weighted.nii.gz \
	   --sid subjectX --sd /output 
```

In conflict with the official ROCm documentation (above), we also needed to add the group render `--group-add render` (in addition to `--group-add video`).

Note, we tested on an AMD Radeon Pro W6600, which is [not officially supported](https://docs.amd.com/en/latest/release/gpu_os_support.html), but setting `HSA_OVERRIDE_GFX_VERSION=10.3.0` [inside docker did the trick](https://en.opensuse.org/SDB:AMD_GPGPU#Using_CUDA_code_with_ZLUDA_and_ROCm):

```bash
docker run --rm --security-opt seccomp=unconfined \
           --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
	   -v /home/user/my_mri_data:/data \
	   -v /home/user/my_fastsurfer_analysis:/output \
	   -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
	   my_fastsurfer:rocm \
	   --t1 /data/subjectX/t1-weighted.nii.gz \
	   --sid subjectX --sd /output 
```

## Build docker image with attestation and provenance

To build a docker image with attestation and provenance, i.e. Software Bill Of Materials (SBOM) information, several requirements have to be met:

1. The image must be built with version v0.11+ of BuildKit (we recommend you [install BuildKit](#buildkit) independent of attestation).
2. You must configure a docker-container builder in buildx (`docker buildx create --use --bootstrap --name fastsurfer-bctx --driver docker-container`). Here, you can add additional configuration options such as safe registries to the builder configuration (add `--config /etc/buildkitd.toml`).
   ```toml
   root = "/path/to/data/for/buildkit"
   [worker.containerd]
     gckeepstorage=9000
     [[worker.containerd.gcpolicy]]
       keepBytes = 512000000
       keepDuration = 172800
       filters = [ "type==source.local", "type==exec.cachemount", "type==source.git.checkout"]
     [[worker.containerd.gcpolicy]]
       all = true
       keepBytes = 1024000000
   # settings to push to a "local", registry with self-signed certificates
   # see for example https://tech.paulcz.net/2016/01/secure-docker-with-tls/ https://github.com/paulczar/omgwtfssl
   [registry."host:5000"]
     ca=["/path/to/registry/ssl/ca.pem"]
     [[registry."landau.dzne.ds:5000".keypair]]
       key="/path/to/registry/ssl/key.pem"
       cert="/path/to/registry/ssl/cert.pem"
   ```
3. Attestation files are not supported by the standard docker image storage driver. Therefore, images cannot be tested locally. 
   There are two solutions to this limitation.
   1. Directly push to the registry: 
      Add `--action push` to the build script (the default is `--action load`, which loads the created image into the current docker context, and for the image name, also add the registry name. For example `... python Docker/build.py ... --attest --action push --tag docker.io/<myaccount>/fastsurfer:latest`.
   2. [Install the containerd image storage driver](https://docs.docker.com/storage/containerd/#enable-containerd-image-store-on-docker-engine), which supports attestation: To implement this on Linux, make sure your docker daemon config file `/etc/docker/daemon.json` includes
      ```json
      {
          "features": {
              "containerd-snapshotter": true
          }
      }
      ```
      Also note, that the image storage location with containerd is not defined by the docker config file `/etc/docker/daemon.json`, but by the containerd config `/etc/containerd/config.toml`, which will likely not exist. You can [create a default config](https://github.com/containerd/containerd/blob/main/docs/getting-started.md#customizing-containerd) file with `containerd config default > /etc/containerd/config.toml`, in this config file edit the `"root"`-entry (default value is `/var/lib/containerd`).  
4. Finally, you can now build the FastSurfer image with `python Docker/build.py ... --attest`. This will add the additional flags to the docker build command.
