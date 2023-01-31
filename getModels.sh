#!/usr/bin/bash
for file in $(find . -name '.zenodo')
do 
	folder=${file::-8}
	tf=$folder/tmp.zip
	wget -nv --content-disposition -i $file -O $tf
	unzip $tf -d $folder
	rm $tf
done
