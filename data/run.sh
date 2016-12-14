wget http://192.168.2.119/matrix/pull_models.tar
tar xvf pull_models.tar
cp pull_models/model_encrypt . 
cp pull_models/model_key.perm .
for file in `ls 0/`
do
    if [ "$file" = "avgface.jpg" -o "$file" = "pedestrian_attribute_tagnames.txt" ];then
        cp 0/$file 1/
    elif [ "$file" = "501.dat" ];then
        cp 0/$file 1/
    elif [ "$file" = "bitri_threshold.txt" ];then
        cp 0/$file 1/
    else
        ./model_encrypt -i 0/"$file" -o 1/"$file"
    fi 
done

rm -rf model_*
rm -rf pull_models*
