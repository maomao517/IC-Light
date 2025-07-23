cd /home/notebook/code/personal/S9059881/IC-Light/imgs/fgs
i=1
for file in *; do
    if [ -f "$file" ]; then
        ext="${file##*.}"
        mv -- "$file" "fg_${i}.${ext}"
        ((i++))
    fi
done