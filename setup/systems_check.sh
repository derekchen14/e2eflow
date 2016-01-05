#! /usr/bin/env bash

sudo apt-get update #the -y means automatically say yes, we want to skip that here for safety purposes
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
# sudo apt-get -y install make python-dev python-setuptools libblas-dev gfortran g++ python-pip python-numpy python-scipy liblapack-dev
# sudo pip install ipython nose
# sudo pip install pandas
#apt-get install nvidia-cuda-toolkit
#sudo pip install --upgrade git+git://github.com/Theano/Theano.git
#sudo pip install --upgrade theano

# setup GPU and CUDA
#sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_6.5-14_amd64.deb
#sudo dpkg -i cuda-repo-ubuntu1404_6.5-14_amd64.deb
#sudo apt-get update
#sudo apt-get install -y cuda # this takes a while
#echo -e "\nexport PATH=/usr/local/cuda-6.5/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64" >> .bashrc
#sudo reboot

# setup theano
cat <<EOF >~/.theanorc
[global]
floatX = float32
device = gpu0
[nvcc]
fastmath = True
EOF

cd ~

python libraries_check.py