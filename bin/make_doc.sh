#!/bin/bash

# Script for generating docs
#
# Author: Exequiel Fuentes Lettura <efulet@gmail.com>

if ! which epydoc >/dev/null
then
  echo "It seems epydoc is not installed on your system"
  exit 1
fi

BINPATH=`dirname $0`

epydoc --html ${BINPATH}/../wdbc/lib -v -o ${BINPATH}/../doc/apidocs \
--name BreastCancerWisconsin --graph all
