import os
import cv2


def sample_data(label_file, root, root_out, interval=5, save_size=1024):
    clip_names = open(label_file, 'r').readlines()
    clip_names = [clip_name.strip() for clip_name in clip_names]
    cnt_all, cnt_all_save = 0, 0
    os.makedirs(root_out, exist_ok=True)
    for i, clip_name in enumerate(clip_names):
        id = clip_name.split()[0]
        vid_path = '{}/{}/{}.mp4'.format(root, id, clip_name)
        cap = cv2.VideoCapture(vid_path)
        frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cnt = 0
        while cnt < frame_len:
            ret, img = cap.read()
            if img is None:
                print('{} ({}/{}): invalid image'.format(vid_path, cnt, int(frame_len)-1))
                cnt += 1
                continue
            if cnt % interval == 0:
                img_path_out = '{}/{}_{}.png'.format(root_out, clip_name, cnt_all_save)
                H, W, C = img.shape
                img = cv2.resize(img, (save_size, save_size))
                cv2.imwrite(img_path_out, img)
                cnt_all_save += 1
            cnt += 1
            cnt_all += 1
            print('\r{}: {}/{} | {}/{} | {}/{}'.format(label_file, i+1, len(clip_names), cnt, frame_len, cnt_all_save, cnt_all), end='')


root = '/media/data1/zhangzjn/TimeLapse/QST/clips'
root_out = '/media/data1/zhangzjn/TimeLapse/QST/samples_stylegan2'
sample_data('train_urls.txt', root=root, root_out='{}/train/1'.format(root_out))
sample_data('test_urls.txt', root=root, root_out='{}/test/1'.format(root_out))
sample_data('val_urls.txt', root=root, root_out='{}/val/1'.format(root_out))
