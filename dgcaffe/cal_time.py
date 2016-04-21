
cnt = [0,0,0,0,0]
max_cnt = [0,0,0,0,0]
min_cnt = [1000,1000,1000,1000,1000]
tot = 0

for line in open('log_tmp'):
    part, time = line.strip().split(':')
    if part.startswith('readling image'):
        cnt[0] += int(time)
        max_cnt[0] = max(max_cnt[0], int(time))
        min_cnt[0] = min(min_cnt[0], int(time))
    elif part.startswith('resize image'):
        cnt[1] += int(time)
        max_cnt[1] = max(max_cnt[1], int(time))
        min_cnt[1] = min(min_cnt[1], int(time))
    elif part.startswith('detection'):
        cnt[2] += int(time)
        max_cnt[2] = max(max_cnt[2], int(time))
        min_cnt[2] = min(min_cnt[2], int(time))
    elif part.startswith('post detection'):
        cnt[3] += int(time)
        max_cnt[3] = max(max_cnt[3], int(time))
        min_cnt[3] = min(min_cnt[3], int(time))
    elif part.startswith('total'):
        cnt[4] += int(time)
        max_cnt[4] = max(max_cnt[4], int(time))
        min_cnt[4] = min(min_cnt[4], int(time))
        tot += 1


print tot
for i in range(len(cnt)):
    print i, 'avg', cnt[i] * 1.0 / tot
    print max_cnt[i]
    print min_cnt[i]


