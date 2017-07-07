#!/bin/bash
# ------------------------------------------------------------------
# [Masaya Ogushi] Install Bazel for docker environments
#
#          library for Unix shell scripts.
#          Description
#              Bazel need for build the tensorflow from source code
#
#          Reference
#              https://bazel.build/versions/master/docs/install.html
#
#
# ------------------------------------------------------------------
# --- Function --------------------------------------------
# -- Body ---------------------------------------------------------
apt-get install -y software-properties-common
apt-get update
add-apt-repository ppa:webupd8team/java
apt-get update
apt-get install -y oracle-java8-installer
apt-get install -y locate
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
apt-get update && apt-get install -y bazel
apt-get update
