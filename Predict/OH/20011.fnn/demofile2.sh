date +"%D %T -> Start testR1"
cd /home/zy/desk/looh210/1.alkane_ch3/data/n6-v210609.1/05.n-1/20220529-fnn/test/795.fnn/testR1
if [ ! -f "all.log" ];then
python3 /home/zy/desk/looh210/1.alkane_ch3/data/n6-v210609.1/05.n-1/20220529-fnn/bin/rate_fnn-v1.5.py -t /home/zy/desk/looh210/1.alkane_ch3/data/n6-v210609.1/05.n-1/test1/CH3_FNN_K1.train.txt -v /home/zy/desk/looh210/1.alkane_ch3/data/n6-v210609.1/05.n-1/test1/CH3_FNN_K1.valid.txt -p "/home/zy/desk/looh210/1.alkane_ch3/data/n6-v210609.1/05.n-1/test1/CH3.mds.test.input.txt,testR1" -L "../pred.list.txt" -l "CH3_testR1" -I 6 -n 500 -r 0.0009 -g 0.45 -H 20 -s 100 -N 1 -w 1 > all.log 2>&1
fi
date +"%D %T -> Finish testR1"

