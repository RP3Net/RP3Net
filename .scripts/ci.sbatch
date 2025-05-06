#!/bin/env bash
#SBATCH --time=30
#SBATCH --mem=64G
#SBATCH -J "RP3Net_CI"
#SBATCH -o "/homes/evgeny/nobackup/log/%x_%j.out"
#SBATCH -e "/homes/evgeny/nobackup/log/%x_%j.err"
#SBATCH --mail-type FAIL
#SBATCH --mail-user evgeny

set -e

. ~/.bashrc
module load python/3.12.9 gcc/11.2.0 cuda/12.2.0

export WORKDIR="/dev/shm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
echo "WORKDIR=${WORKDIR}"
mkdir "${WORKDIR}"
cd "${WORKDIR}"

git clone ebi:~/git/RP3Net.git
cd RP3Net
git switch _gh_workflow_ebi_tests

GIT_DIFF_OUT=$(git --no-pager diff --name-only origin/develop); GIT_DIFF_RET=$?;
if [ "${GIT_DIFF_RET}" != "0" ]; then
    exit ${GIT_DIFF_RET}
fi
export GIT_DIFF_OUT
echo "GIT_DIFF_OUT=${GIT_DIFF_OUT}"

if [ "${GIT_DIFF_OUT}" != "" ]; then
    git rebase origin/develop

    python -m venv --clear "${WORKDIR}/venv"
    . "${WORKDIR}/venv/bin/activate"
    which python
    python --version

    pip install --upgrade pip
    pip install '.[training]'

    python -m unittest discover -v -s ./tests_ebi -p '*test.py'
    python -m unittest discover -v -s ./tests -p '*test.py'

    git push

    echo 'SUCCESS'

else
    echo "No git diff so skipping tests"
fi

sleep 10
