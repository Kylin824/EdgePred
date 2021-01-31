import pandas as pd
import time
import math


"""

将聚类后的poi按时间段分割

c1_poi_cluster.csv  => c1_poi_span.csv

"""

dataset = pd.read_csv('c1_poi_cluster.csv')

value = dataset.values

length = len(value)

span = 60 * 10  # 10分钟记录一次


cols = ['daytype', 'whatday', 'poi', 'dt','dt_scale']  #
total_list = []


last_second = 0

for i in range(length):

    day_type = value[i][0]
    what_day = value[i][1]
    poi = value[i][2]

    zero_time = str(value[i][3]) + ' 00:00:00'
    start_time = str(value[i][3]) + ' ' + str(value[i][4])
    end_time = str(value[i][5]) + ' ' + str(value[i][6])
    final_time = str(value[i][3]) + ' 23:59:59'

    zero_timestamp = time.mktime(time.strptime(zero_time, "%Y-%m-%d %H:%M:%S"))   # 0点时间戳
    zero_second = 0
    start_second = time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S")) - zero_timestamp  #
    end_second = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S")) - zero_timestamp + 1
    final_second = time.mktime(time.strptime(final_time, "%Y-%m-%d %H:%M:%S")) - zero_timestamp + 1  # 12点时间戳

    span_num = math.ceil((end_second - start_second) / span)

    for j in range(span_num):

        dt_second = start_second - (start_second % span) + j * span

        if dt_second == last_second:
            continue

        dt_timestamp = dt_second + zero_timestamp
        dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dt_timestamp))
        dt_scale = dt_second / final_second

        total_list.append([day_type, what_day, poi, dt, dt_scale])
        last_second = dt_second

for i in range(len(total_list)):
    print(total_list[i])

df = pd.DataFrame(columns=cols, data=total_list)
df.to_csv('./c1_poi_span.csv', index=0)

