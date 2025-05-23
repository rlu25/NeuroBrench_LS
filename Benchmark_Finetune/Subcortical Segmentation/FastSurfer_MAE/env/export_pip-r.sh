#!/bin/bash

# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function usage()
{
cat <<EOF
export_pip-r.sh <requirements file> <docker image/singularity image file>

Will generate requirements files from docker/singularity images to distribute with
the FastSurfer repository.

examples:
export_pip-r.sh $FASTSURFER_HOME/requirements.txt fastsurfer:dev
export_pip-r.sh $FASTSURFER_HOME/requirements.cpu.txt fastsurfer:dev

Created 26-11-2023, David Kügler, Image Analysis Lab,
German Center for Neurodegenerative Diseases (DZNE), Bonn
EOF
}

if [ "$#" != 2 ]
then
  echo "This scripts expects 2 parameters, got $#"
  exit 1
fi

echo "Exporting versions from $2..."

{
  echo "#"
  echo "# This file is autogenerated by $USER from $2"
  echo "# by the following command from FastSurfer:"
  echo "#"
  echo "#    $0 $*"
  echo "#"
} > $1

pip_cmd="python --version && pip list --format=freeze --no-color --disable-pip-version-check --no-input --no-cache-dir"
if [ "${2/#.sif}" != "$2" ]
then
  # singularity
  cmd=("singularity" "exec" "$2" "/bin/bash" -c "$pip_cmd")
  clean_cmd="singularity exec $2 /bin/bash -c '$pip_cmd'"
else
  # docker
  clean_cmd="docker run --rm -u <user_id>:<group_id> --entrypoint /bin/bash $2 -c '$pip_cmd'"
  cmd=("docker" "run" --rm -it -u "$(id -u):$(id -g)" --entrypoint /bin/bash "$2" -c "$pip_cmd")
fi
{
  echo "# Which ran the following command:"
  echo "#    $clean_cmd"
  echo "#"
} >> $1
echo "Running '${cmd[*]}' to get $1"
out=$("${cmd[@]}")
hardware=$(echo "$out" | grep "torch==" | cut -d"+" -f2 | tr -d '\n')
pyversion=$(echo "$out" | head -n 1 | cut -d" " -f2)
{
  echo "#"
  echo "# Image was configured for $hardware using python version $pyversion"
  echo "#"
  echo "--extra-index-url https://download.pytorch.org/whl/$hardware"
  echo ""
  # first line is python version and gets commented here
  echo "# $out"
} >> $1
