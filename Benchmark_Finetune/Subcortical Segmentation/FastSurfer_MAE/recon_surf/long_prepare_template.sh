#!/bin/bash

# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

###################################################################################################
#
#
#  FastSurfer Longitudinal Base Template Creation
# 
# 
#  This script is part of the FastSurfer longitudinal pipeline. It runs a few pre-processing steps
#  to setup a base template for processing with a slightly modified version of run_fastsurfer.
# 
#  1. It obtains a brainmask on all time points for a specific subject
#  2. It uses mri_robust_template to co-register all time points into a mid-space
#     and sets up the base (subject template) directory with the aligned images the median image
#     and the forward and backward rigid transformations
# 
#  Following this script, one would process the median (subject template) image (-base) and then 
#  initialize each time point (-long) with information form this template for the surface module.
# 
#  Potential future things:
#   - Check if a bias field correction before the registration is helpful.
#   - Check if single view FastSurferVINN network is sufficient or if multi-view helps.
#   - Check if centroid based alignment of the segmentation helps for initializing robust_template.
#   - Maybe use intersection of tp masks as brainmask for base, as done in FreeSurfer.
#   - Add flag for adding a new time point to an existing base/template.
#
#  FreeSurfer requirements: 
#  mri_convert, mri_robust_template, mri_mask, mri_concatenate_lta, make_upright
#
###################################################################################################


# Set default values for arguments
if [[ -z "${BASH_SOURCE[0]}" ]]; then THIS_SCRIPT="$0"
else THIS_SCRIPT="${BASH_SOURCE[0]}"
fi
if [[ -z "$FASTSURFER_HOME" ]]
then
  FASTSURFER_HOME=$(cd "$(dirname "$THIS_SCRIPT")" &> /dev/null && pwd)
  echo "Setting ENV variable FASTSURFER_HOME to script directory ${FASTSURFER_HOME}. "
  echo "Change via environment to location of your choice if this is undesired (export FASTSURFER_HOME=/dir/to/FastSurfer)"
  export FASTSURFER_HOME
fi

# Paths
fastsurfercnndir="$FASTSURFER_HOME/FastSurferCNN"
reconsurfdir="$FASTSURFER_HOME/recon_surf"

# some fixed variables
extension=".nii.gz" # this script already works completely with nifti except for the final <TID>/mri/orig.mgz output
interpol="cubic"    # for the final interpolation of all time points in median image
robust_template_avg_arg=1  # median for template creation (if more than 1 time point)

# default arguments
python="python3.10 -s" # avoid user-directory package inclusion
sd="$SUBJECTS_DIR"

# init variables that need to be passed
tid=""
tpids=()
t1s=()

source "${reconsurfdir}/functions.sh"

function usage()
{
cat << EOF

Usage: long_prepare_template.sh --tid <sid> --t1s <T1_1> <T1_2> .. \\
                                --tpids <ID1> <ID2> .. \\
                                --sd <sdir> [OPTIONS]

long_prepare_template.sh takes a list of T1 full head images and creates:
     (i)   a template/base subject directory: <SUBJECTS_DIR>/<TID>
     (ii)  co-registered images for all time points:
           <TID>/long-inputs/<tpid>/long_conform.nii.gz
     (iii) median image as template for this subject <TID>/mri/orig.mgz

FLAGS:

  --tid <templateID>      ID for subject template/base directory inside
                            \$SUBJECTS_DIR to be created"
  --t1s <T1_1> <T1_2> ..  T1 full head inputs for each time point (do not need
                            to be bias corrected). Requires ABSOLUTE paths!
  --tpids <ID1> >ID2> ..  IDs for future time points directories inside
                            \$SUBJECTS_DIR to be created later (during --long)
  --sd  <subjects_dir>    Output directory \$SUBJECTS_DIR (or pass via env var)
  --vox_size <0.7-1|min>  Forces processing at a specific voxel size.
                            If a number between 0.7 and 1 is specified (below
                            is experimental) the T1w image is conformed to
                            that voxel size and processed.
                            If "min" is specified (default), the voxel size is
                            read from the size of the minimal voxel size
                            (smallest per-direction voxel size) in the T1w
                            image:
                              If the minimal voxel size is bigger than 0.98mm,
                                the image is conformed to 1mm isometric.
                              If the minimal voxel size is smaller or equal to
                                0.98mm, the T1w image will be conformed to
                                isometric voxels of that voxel size.
                            The voxel size (whether set manually or derived)
                            determines whether the surfaces are processed with
                            highres options (below 1mm) or not.
  -h --help                Print Help

Resource Options:
  --device                Set device on which inference should be run ("cpu" for
                            CPU, "cuda" for Nvidia GPU, or pass specific device,
                            e.g. cuda:1), default check GPU and then CPU
  --viewagg_device <str>  Define where the view aggregation should be run on.
                            Can be "auto" or a device (see --device). By default,
                            the program checks if you have enough memory to run
                            the view aggregation on the gpu. The total memory is
                            considered for this decision. If this fails, or you
                            actively overwrote the check with setting with "cpu"
                            view agg is run on the cpu. Equivalently, if you
                            pass a different device, view agg will be run on that
                            device (no memory check will be done).
  --batch <batch_size>    Batch size for inference. Default: 1
  --py <python_cmd>       Command for python, used in both pipelines.
                            Default: "$python"
                            (-s: do no search for packages in home directory)

EOF
}

# PRINT USAGE if called without params
if [[ $# -eq 0 ]]
then
  usage
  exit
fi

# PARSE Command line
inputargs=("$@")
POSITIONAL_FASTSURFER=()
run_pred_flags=()
i=0
while [[ $# -gt 0 ]]
do
# make key lowercase
arg="$1"
key=$(echo "$arg" | tr '[:upper:]' '[:lower:]')

shift # past argument
case $key in
  --tid) tid="$1" ; shift ;;
  --tpids) while [[ $# -gt 0 ]] && [[ $1 != -* ]] ; do tpids+=("$1") ; shift ; done ;;
  --t1s) while [[ $# -gt 0 ]] && [[ $1 != -* ]] ; do t1s+=("$1") ; shift ; done ;;
  --sd) sd="$1" ; export SUBJECTS_DIR="$1" ; shift  ;;
  # these flags are passed through to run_prediction.py
  --vox_size|--device|--viewagg_device|--conform_to_1mm_threshold) run_pred_flags+=("$key" "$1") ; shift ;;
  --threads|--threads_seg) run_pred_flags+=("--threads" "$1") ; shift ;;
  --batch) run_pred_flags+=("--batch_size" "$1") ; shift ;;
  # these known arguments get ignored
  --aseg_name|--conformed_name|--asegdkt_segfile|--brainmask_name|--seg_log|--qc_log|--parallel|--threads_surf) shift ;;
  --no_cereb|--no_hypothal|--no_biasfield|--3t) shift ;;
  --async_io) ;;
  --fs_license) export FS_LICENSE="$1" ; shift ;;
  --remove_suffix) echo "ERROR: The --remove_suffix option is not supported by long_prepare_template.sh" ; exit 1 ;;
  -h|--help) usage ; exit ;;
  --py) python="$1" ; shift ;;
  *)    # unknown options also get ignored, but also print warnings
    POSITIONAL_FASTSURFER[i]="$arg"
    i=$((i + 1))
    ;;
  #*)    # unknown option
  #  # if not empty arguments, error & exit
  #  if [[ "$key" != "" ]] ; then echo "ERROR: Flag '$key' unrecognized." ;  exit 1 ; fi
  #  ;;
esac
done


################################## CHECKS ##############################

if [[ "${#POSITIONAL_FASTSURFER[@]}" -gt 0 ]]
then
  echo "WARNING: The arguments ${POSITIONAL_FASTSURFER[*]}"
  echo "  are not recognized and therefore ignored in this (sub-)script!"
fi

if [ "${#t1s[@]}" -lt 1 ]
then
  echo "ERROR: You must supply T1 inputs (full head) via --t1s!"
  exit 1
fi

if [ "${#tpids[@]}" -lt 1 ]
then
  echo "ERROR: You must supply time points ids via --tpids!"
  exit 1
fi

if [ -z "$tid" ]
then
  echo "ERROR: You must supply subject template name via --tid!"
  exit 1
fi

# check that t1s list is same length as tpids
if [ "${#tpids[@]}" -ne "${#t1s[@]}" ]
then
  echo "ERROR: The length of tpids must equal t1s!"
  exit 1
fi

# check that SUBJECTS_DIR exists
check_create_subjects_dir_properties "$SUBJECTS_DIR"

################################## SETUP and LOGFILE ##############################


# Setup Base/Template Directory and Log file
LF="$SUBJECTS_DIR/$tid/scripts/long_prepare_template.log"
mkdir -p "$(dirname "$LF")"

export PYTHONPATH
PYTHONPATH="$FASTSURFER_HOME$([[ -n "$PYTHONPATH" ]] && echo ":$PYTHONPATH")"

## make sure +eo are unset
set +eo > /dev/null

if [[ -f "$LF" ]]; then log_existed="true" ; else log_existed="false" ; fi

version_args=()
if [[ -f "$FASTSURFER_HOME/BUILD.info" ]]
then
  version_args=(--build_cache "$FASTSURFER_HOME/BUILD.info" --prefer_cache)
fi

VERSION=$($python "$FASTSURFER_HOME/FastSurferCNN/version.py" "${version_args[@]}")
code="$?"
if [[ "$code" != 0 ]] ; then echo "ERROR: Getting the version failed (code=$code), terminating..." ; exit 1 ; fi
echo "Version: $VERSION" | tee -a "$LF"
echo "Log file for long_prepare_template" >> "$LF"
{
  date 2>&1
  echo ""
  echo ""
  echo "export SUBJECTS_DIR=$SUBJECTS_DIR"
  echo "cd `pwd`"
  echo "$0 ${inputargs[*]}"
  echo ""
  cat "$FREESURFER_HOME/build-stamp.txt" 2>&1
  uname -a  2>&1
} | tee -a "$LF"


### IF THE SCRIPT GETS TERMINATED, ADD A MESSAGE
# shellcheck disable=SC2064
trap "{ echo \"long_prepare_template.sh terminated via signal at \$(date -R)!\" >> \"$LF\" ; }" SIGINT SIGTERM


# check that all t1s exist and that geo is the same (after log setup to keep this info in log file)
geodiff=""
for s in "${t1s[@]}"
do
  # check if input exist
  if [[ ! -f "$s" ]]
  then
    echo "ERROR: Input T1 $s does not exist!" | tee -a "$LF"
    exit 1
  fi
  # check if geometry differs across time
  if [[ "$s" != "${t1s[0]}" ]]
  then
    cmda=(mri_diff --notallow-pix --notallow-geo "$s" "${t1s[0]}" --res-thresh "0.000001")
    difftext=$("${cmda[@]}")
    retcode=${PIPESTATUS[0]}
    if [[ "$retcode" != 0 ]] ; then geodiff+="Comparing $s and ${t1s[0]} (code $retcode):\n$difftext\n" ; fi
  fi
done
if [[ -n "$geodiff" ]]
then
  {
    echo " "
    echo "*******************************************************************************" 
    echo "WARNING: Image parameters differ across time, maybe due to acquisition changes?"
    echo "         Consistent changes in, e.g., resolution can potentially bias a "
    echo "         longitudinal study! You can check image parameters by running mri_info"
    echo "         on each input image."
    echo "*******************************************************************************"
    echo "$geodiff"
    # if we are in a terminal (stdin is a terminal), wait 10 seconds
    if [[ -t 0 ]] ; then echo "    Will continue in 10 seconds... (Abort with Ctrl+C)" ; fi
    echo ""
    echo "*******************************************************************************"
    echo " "
  } | tee -a "$LF"
  if [[ -t 0 ]] ; then sleep 10 ; fi
fi


################################### MASK INPUTS ###################################


# here we prepare images for base creation (image registration) below
# i)   either do nothing (maybe we try that)
# ii)  or compute brainmask (e.g. via single or multi view segmentation) <--- !!!
# iii) or even compute bias field removed images (one-shot bias field removal)

{
  echo " "
  echo "================== Creating Brainmask per TP ========================="
  echo " "
} | tee -a "$LF"

for ((i=0;i<${#tpids[@]};++i)); do
  #printf "%s with T1 %s\n" "${tpids[i]}" "${t1s[i]}"
  echo "${tpids[i]} with T1 ${t1s[i]}" | tee -a "$LF"
  mdir="$SUBJECTS_DIR/$tid/long-inputs/${tpids[i]}"
  mkdir -p "$mdir"
  # Import (copy) raw inputs (convert to extension format)
  t1input=$mdir/cross_input${extension}
  cmd="mri_convert ${t1s[i]} $t1input"
  RunIt "$cmd" "$LF"
  
  # conform !!!!!!! should we conform to some common value, determined from all time points?? !!!!!!
  # this is relevant if input resolutions different (which they should not), currently conform min may not work as expected
  #conform="$mdir/T1_conform${extension}"
  #cmd="mri_convert -c ${t1s[i]} $conform"
  #cmd="$python $fastsurfercnndir/data_loader/conform.py -i ${t1s[i]} -o $conform --vox_size $vox_size --dtype any --verbose"
  #RunIt "$cmd" $LF

  # segment image
  # with the goal to create brainmask for mainly registration
  # (here we can probably only use one network)
  asegdkt_segfile="$mdir/cross_aparc+aseg.orig${extension}"
  conformed_name="$mdir/cross_conform${extension}"
  mask_name="$mdir/cross_mask${extension}"
  aseg_segfile="$mdir/cross_aseg.auto_noCCseg${extension}"
  seg_log="/dev/null"
  cmda=($python "$fastsurfercnndir/run_prediction.py" --t1 "$t1input" --async_io
         --asegdkt_segfile "$asegdkt_segfile" --conformed_name "$conformed_name"
         --brainmask_name "$mask_name" --aseg_name "$aseg_segfile" --sid "${tpids[i]}"
         --seg_log "$seg_log" "${run_pred_flags[@]}")
  run_it "$LF" "${cmda[@]}"

  # remove mri subdirectory (run_prediction creates 001 there)
  cmda=(rm -rf "$mdir/mri")
  run_it "$LF" "${cmda[@]}"
  
  # mask is binary, we need to use on conformed image:
  cmda=(mri_mask "$conformed_name" "$mask_name" "$mdir/cross_brainmask${extension}")
  run_it "$LF" "${cmda[@]}"
done

# skip intensity normalization or bias field removal for now




#################################### CO-REGISTER INPUTS ############################################

{
  echo " "
  echo "================== Co-registering TPs ========================="
  echo " "
} | tee -a "$LF"

# create a file with all time points names
# this cannot be "base-tps" else recon-surf (and inside recon-all) will fail
BaseSubjsListFname="$SUBJECTS_DIR/$tid/base-tps.fastsurfer"
rm -f "${BaseSubjsListFname}"
mkdir -p "$SUBJECTS_DIR/$tid/mri/transforms"
subjInVols=()
normInVols=()
ltaXforms=()

for s in "${tpids[@]}"
do
  echo "$s" | tee -a "${BaseSubjsListFname}"
  mdir="$SUBJECTS_DIR/$tid/long-inputs/${s}"
  invol="$mdir/cross_conform${extension}"
  subjInVols+=("$invol")
  normvol="$mdir/cross_brainmask${extension}"
  normInVols+=("$normvol")
  ltaname="${s}_to_${tid}.lta"
  ltaXforms+=("${SUBJECTS_DIR}/$tid/mri/transforms/${ltaname}")
done


if [ ${#tpids[@]} == 1 ]
then
  # if only a single time point, create fake 'base' by making the image upright
  # this assures that also subjects with a single time point get processed as the other
  # subjects in the longitudinal stream

  # 1. make the norm upright (base space)
  cmd="make_upright ${normInVols[0]} \
       ${SUBJECTS_DIR}/$tid/mri/base_brainmask${extension} ${ltaXforms[0]}"
  RunIt "$cmd" "$LF"

  # 2. create the upright orig volume
  cmd="mri_convert -rt cubic \
       -at ${ltaXforms[0]} ${subjInVols[0]} ${SUBJECTS_DIR}/$tid/mri/orig.mgz"
  RunIt "$cmd" "$LF"

else #more than 1 time point:


  # create the 'mean/median' norm volume:
  cmd="mri_robust_template --mov ${normInVols[*]}"
  cmd="$cmd --lta ${ltaXforms[*]}"
  cmd="$cmd --template ${SUBJECTS_DIR}/$tid/mri/base_brainmask${extension}"
  cmd="$cmd --average ${robust_template_avg_arg}"
  cmd="$cmd --sat 4.685"
  RunIt "$cmd" "$LF"

  # create the 'mean/median' input (orig) volume:
  cmd="mri_robust_template --mov ${subjInVols[*]}"
  cmd="$cmd --average ${robust_template_avg_arg}"
  cmd="$cmd --ixforms ${ltaXforms[*]}"
  cmd="$cmd --noit"
  t1=${SUBJECTS_DIR}/$tid/mri/orig.mgz
  cmd="$cmd --template $t1"
  RunIt "$cmd" "$LF"

fi # more than one time point

# now create the inverse transforms
#cd $subjdir/mri/transforms > /dev/null
#$PWD |& tee -a $LF
odir=${SUBJECTS_DIR}/$tid/mri/transforms
for s in "${tpids[@]}"
do
  cmd="mri_concatenate_lta -invert1"
  cmd="$cmd $odir/${s}_to_${tid}.lta"
  cmd="$cmd identity.nofile"
  cmd="$cmd $odir/${tid}_to_${s}.lta"
  RunIt "$cmd" "$LF"
done

# finally map inputs to template space for each time point
for ((i=0;i<${#tpids[@]};++i))
do
  mdir="$SUBJECTS_DIR/$tid/long-inputs/${tpids[i]}"
  # map orig to base space
  cmd="mri_convert -at ${ltaXforms[$i]} -rt $interpol $mdir/cross_input${extension} $mdir/long_conform${extension}"
  RunIt "$cmd" "$LF"
done


