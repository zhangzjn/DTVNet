import os
import cv2
import csv
import shutil

def make_dataset(dir):

    n = 0
    f = open('/home/lyc/data/motion_prediction/sky_timelapse/not_con.txt', 'a')
    for target in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir, target)) == True:
            subfolder_path = os.path.join(dir, target)
            for subsubfold in sorted(os.listdir(subfolder_path)):
                if os.path.isdir(os.path.join(subfolder_path, subsubfold)):
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold)
                    fi_num_former, fi_num = 0, 0
                    for i, fi in enumerate(sorted(os.listdir(subsubfolder_path))):
                        if i == 0:
                            fi_num = int(fi.split('_frames_')[-1].split('.')[0])
                            fi_num_former = fi_num
                            continue
                        fi_num = int(fi.split('_frames_')[-1].split('.')[0])
                        if fi_num != (fi_num_former + 1):
                            n += 1
                            strtxt = fi_former + fi
                            print('{}: {}'.format(n, strtxt))
                            f.write(strtxt + '\n')
                        fi_num_former = fi_num
                        fi_former = fi

def make_KTH(root):
    txtfile = open(os.path.join(root, 'train_data_list_trimmed.txt'), 'r')
    videopath = os.path.join(root, 'video')
    imagepath = os.path.join(root, 'image')

    name = []
    start = []
    end = []
    for line in txtfile.readlines():
        if len(line.split()) == 0:
            continue
        n, s, e = line.split()
        name.append(n)
        start.append(int(s))
        end.append(int(e))
    txtfile.close()

    num = 1
    for i in range(0, len(name), 1):
        if i + 1 <= len(name) - 1:

            if name[i] == name[i+1]:
                num += 1
                if i + 1 != len(name) - 1:
                    continue

        video_name = os.path.join(videopath, name[i] + '_uncomp.avi')
        save_path = os.path.join(imagepath, 'train', name[i].split('_')[1])
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path_sub = os.path.join(save_path, name[i])
        if not os.path.exists(save_path_sub):
            os.mkdir(save_path_sub)

        cap = cv2.VideoCapture(video_name)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            for j in range(num):

                if count >= start[i - (num - 1) + j] and count <= end[i - (num - 1)+ j]:
                    save_path_sub_sub = os.path.join(save_path_sub, str(j))
                    if not os.path.exists(save_path_sub_sub):
                        os.mkdir(save_path_sub_sub)
                    frame = cv2.resize(frame, (128, 128))
                    cv2.imwrite(os.path.join(save_path_sub_sub, name[i] + '_{:04d}.jpg'.format(count)), frame)

        num = 1
        print('\r{}/{}:{}'.format(i, len(name), name[i]), end='')

def collect_samples(root):
    import os
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()

    samples = []
    for cls in classes:
        folder_path = os.path.join(root, cls)
        for subfold in sorted(os.listdir(folder_path)):
            if os.path.isdir(os.path.join(folder_path, subfold)):
                subfolder_path = os.path.join(folder_path, subfold)
                for subsubfold in sorted(os.listdir(subfolder_path)):
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold)
                    files = sorted(os.listdir(subsubfolder_path))
                    for i in range(1, len(files) - 1, 1):
                        if i + 1 > (len(files) - 1):
                            continue
                        file_name1 = files[i]
                        file_name2 = files[i+1]
                        file_path1 = os.path.join(subsubfolder_path, file_name1)
                        file_path2 = os.path.join(subsubfolder_path, file_name2)
                        s = {'imgs': [file_path1, file_path2]}
                        samples.append(s)

    return samples

def SSIM_PSNR(root):
    src = os.path.join(root, 'real_video')
    dst = os.path.join(root, 'fake_video')
    src_file = sorted(os.listdir(src))
    dst_file = sorted(os.listdir(dst))
    save_root = os.path.join(root, 'log_ssim')

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for i in range(len(src_file)):
        src_path = os.path.join(src, src_file[i])
        dst_path = os.path.join(dst, dst_file[i])
        save_path = os.path.join(save_root, '{}.log'.format(src_file[i]))
        cmd = ('ffmpeg -i ' + src_path + ' -i ' + dst_path + ' -lavfi ssim="stats_file=' + save_path + '" -f null -')
        os.system(cmd)

def AnalyzePSNR(root):
    save_root = os.path.join(root, 'log_psnr')
    l = len(os.listdir(save_root))
    s_s = 0.0
    with open(root + "psnr.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        line = list(map(str, list(range(32))))
        datarow = ['name'] + line
        writer.writerow(datarow)

    for name in sorted(os.listdir(save_root)):
        save_path = os.path.join(save_root, name)
        with open(save_path, 'r') as f:
            lines = f.readlines()
        s = [0.0] * 32

        for i, line in enumerate(lines):
            s[i] += float(line.split(' ')[5].split(':')[-1])
            s_s += float(line.split(' ')[5].split(':')[-1])

        with open(root + "psnr.csv", "a+") as csvfile:
            writer = csv.writer(csvfile)
            datarow = [name] + list(map(str, s))
            writer.writerow(datarow)

        # print(r'{}:{}'.format(name, avg), end='')
    print('\nfinal avg PSNR of all videos is :{}'.format(s_s / (l * 32)))

def AnalyzeSSIM(root):
    save_root = os.path.join(root, 'log_ssim')
    l = len(os.listdir(save_root))
    s_s = 0.0
    with open(root + "ssim.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        line = list(map(str, list(range(32))))
        datarow = ['name'] + line
        writer.writerow(datarow)

    for name in sorted(os.listdir(save_root)):
        save_path = os.path.join(save_root, name)
        with open(save_path, 'r') as f:
            lines = f.readlines()
        s = [0.0] * 32

        for i, line in enumerate(lines):
            s[i] += float(line.split(' ')[4].split(':')[-1])
            s_s += float(line.split(' ')[4].split(':')[-1])

        with open(root + "ssim.csv", "a+") as csvfile:
            writer = csv.writer(csvfile)
            datarow = [name] + list(map(str, s))
            writer.writerow(datarow)
        # print(r'{}:{}'.format(name, avg), end='')
    print('\nfinal avg SSIM of all videos is :{}'.format(s_s / (l * 32)))

def cmp(root):
    import imageio
    save1 = '/home/lyc/ckp/videoprediction/test/test_30_r_1/'
    save2 = '/home/lyc/ckp/videoprediction/test/test_30_r_2/'
    for name in os.listdir(root):
        cap = cv2.VideoCapture(root + name)
        out1 = imageio.get_writer(save1 + name, fps=10.0)
        out2 = imageio.get_writer(save2 + name, fps=10.0)
        count = 0
        while True:
            ret, frame = cap.read()
            count += 1
            if not ret:
                break
            if count == 1:
                frame_first = frame
            out1.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out2.append_data(cv2.cvtColor(frame_first, cv2.COLOR_BGR2RGB))
        out1.close()
        out2.close()

def select_frame(root):
    src = os.path.join(root, 'videos')
    dst = os.path.join(root, 'video')
    num = 0
    for i in os.listdir(src):
        s_src = os.path.join(src, i)
        for j in os.listdir(s_src):
            ss_src = os.path.join(s_src, j)
            for k in os.listdir(ss_src):
                sss_src = os.path.join(ss_src, k)
                folder = sorted(os.listdir(sss_src))
                for name in folder:
                    if not os.path.exists(os.path.join(dst, name)):
                        num = num + 1
                        shutil.copytree(os.path.join(sss_src, name), os.path.join(dst, name))
                        print('\rfile: {}:{}'.format(str(num), name), end='')


if __name__ == '__main__':
    # make_KTH('/home/lyc/data/motion_prediction/KTH/')
    # make_dataset('/home/lyc/data/motion_prediction/sky_timelapse/sky_train/')
    # collect_samples('/home/lyc/data/motion_prediction/KTH/image/train/')
    # AnalyzeSSIM('/home/lyc/code/gan/mocogan/logs/metric/')
    AnalyzeSSIM('/home/lyc/code/gan/mocogan/logs/test10/vid179/')

