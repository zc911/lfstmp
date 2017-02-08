import sys

#in_path = "/home/zz/code/FaceVeri/dgface/top1_score.log"

def top1_perform(in_path):
    step1_score_mean, step2_score_mean, num_lines, num_p, num_n = 0, 0, 0, 0, 0
    with open(in_path, 'r') as f:
        try:
            while True:
                line = f.next().strip()
                img_path, map_score, step1_score, step2_score = line.split('\t')
                step1_score_mean = step1_score_mean + float(step1_score)
                step2_score_mean = step2_score_mean + float(step2_score)
                num_lines = num_lines + 1
                if float(map_score) == 1:
                    num_p = num_p + 1
                else:
                    num_n = num_n + 1
        except StopIteration:
            pass
    step1_score_mean = step1_score_mean / num_lines;
    step2_score_mean = step2_score_mean / num_lines;

    high_thresh = 0.;
    low_thresh = 0.;
    map_thresh = 0.95;
    step2_win_pos = 0;
    step2_win_neg = 0;
    step1_win_pos = 0;
    step1_win_neg = 0;
    s1_pos, s1_neg, s2_pos, s2_neg = 0, 0, 0, 0
    comb_pos = 0;
    comb_neg = 0;
    with open(in_path, 'r') as f:
        try:
            while True:
                flag = 1;
                line = f.next().strip()
                img_path, map_score, step1_score, step2_score = line.split('\t')
                s1 = float(step1_score) - step1_score_mean
                s2 = float(step2_score) - step2_score_mean
                m = float(map_score)
                if m >= map_thresh and s2 > s1 and s2 > high_thresh:
                    flag = 2
                    step2_win_pos = step2_win_pos + 1
                elif m < map_thresh and s2 < s1 and s2 < low_thresh:
                    flag = 2
                    step2_win_neg = step2_win_neg + 1
                elif m >= map_thresh and s2 < s1 and s1 > high_thresh:
                    step1_win_pos = step1_win_pos + 1
                elif m < map_thresh and s2 > s1 and s1 < low_thresh:
                    step1_win_neg = step1_win_neg + 1
                else:
                    flag = 0;

                if m >= map_thresh:
                    if s1 > high_thresh:
                        s1_pos = s1_pos + 1
                    if s2 > high_thresh:
                        s2_pos = s2_pos + 1
                    if s1 > high_thresh and s2 > high_thresh:
                        comb_pos = comb_pos + 1
                elif m < map_thresh:
                    if s1 < low_thresh:
                        s1_neg = s1_neg + 1
                    if s2 < low_thresh:
                        s2_neg = s2_neg + 1
                    if s1 < low_thresh or s2 < low_thresh:
                        comb_neg = comb_neg + 1

                    print img_path + '\t' + map_score + '\t' + str(s1) + '\t' + str(s2) + '\t' + str(flag)
        except StopIteration:
            pass

    print "queries: " + str(num_lines) + "\t" + "corr: " + str(num_p) + "\t" + "wrong: " + str(num_n)
    print "s2_mean: " + str(step2_score_mean) + "\t" + "s1_mean: " + str(step1_score_mean)
    print "high_thresh: " + str(high_thresh) + "\t" + "low_thresh: " + str(low_thresh)
    print "s2_win_pos: " + str(step2_win_pos) + "\t" + "s2_win_neg: " + str(step2_win_neg)
    print "s1_win_pos: " + str(step1_win_pos) + "\t" + "s1_win_neg: " + str(step1_win_neg)
    print "==================================="
    print "sc_pos: " + str(comb_pos) + "\t" + "sc_neg: " + str(comb_neg)
    print "s2_pos: " + str(s2_pos) + "\t" + "s2_neg: " + str(s2_neg)
    print "s1_pos: " + str(s1_pos) + "\t" + "s1_neg: " + str(s1_neg)

if __name__ == "__main__":
    in_path = sys.argv[1] 
    top1_perform(in_path)
