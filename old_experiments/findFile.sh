if [ $# -lt 2 ]; then
    echo "use: $0 file dir"
    exit 1
fi
if [ ! -f $1 ]; then
    echo "file $1 don't exists"
    exit 2
fi
if [ ! -d $2 ]; then
    echo "dir $2 don't exists"
    exit 3
fi

md5First="$(dd if=$1 bs=512 count=1 status=none | md5sum | rev | cut -c4- | rev)"
md5FirstLong="$(dd if=$1 status=none | md5sum | rev | cut -c4- | rev)"

files=`find $2 -type f`

for file in $files; do
    md5Second="$(dd if=$file bs=512 count=1 status=none | md5sum | rev | cut -c4- | rev)"
    if [ "$md5First" = "$md5Second" ]; then
        md5SecondLong="$(dd if=$file status=none | md5sum | rev | cut -c4- | rev)"
        if [ "$md5FirstLong" = "$md5SecondLong" ]; then
            echo "$1 and $file are equal"
        fi
    fi
done
