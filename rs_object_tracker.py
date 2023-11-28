
from realsense import *
import numpy as np
import cv2


CUDA = torch.cuda.is_available()

# HISTCMP OPTIONS
#
# cv2.HISTCMP_CORREL
# cv2.HISTCMP_CHISQR
# cv2.HISTCMP_INTERSECT
# cv2.HISTCMP_BHATTACHARYYA
HISTCMP = cv2.HISTCMP_CORREL

# 이미지 제거 결정 프레임 수
img_drop_decision_frame = 10

# 이미지 후순위 갱신 프레임
late_img_renewal_frame = 5

# Option 1
#HISTCMP_2 = None
#이미지 비교 신규 객체 추가
#object_limitations = 0.75

# Option 2
HISTCMP_2 = cv2.HISTCMP_INTERSECT
object_limitations = 0.175

#비디오 경로
#videoURI = 'T1.mp4'
videoURI = '0819_t4.mp4'

#욜로 설정파일
cfgfile = "cfg/yolov3.cfg"

#욜로 가중치 파일
weightsfile = "cfg/yolov3.weights"

#욜로 객체 신뢰도
confidence = 0.5

#욜로 함수?
nms_thesh = 0.4

#박스 그리는 플레그
is_draw = True

#보여주는 플레그
is_show = True

#이미지 리사이즈
is_resize = True
resize_width = 640
resize_height = 480

#이미지 작성
is_img_write = True
img_write_frame_szie = (640, 480)
img_write_fps = 30

# 리셋 플레그
reset_flag = False
reset_flag_sender = 0


def get_img_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def model_build(yolo_cfg, weight, confidence, nms_thesh):
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    model.net_info["height"] = 160
    inp_dim = int(model.net_info["height"])
    if CUDA:
        model.cuda()
    model.eval()
    return model, inp_dim


def img_cut_by_box(img, box):
    return img[box[1]:box[3], box[0]:box[2]]


def get_cut_imgs_by_boxs(img, boxs):
    output = []
    for box in boxs:
        output.append(img_cut_by_box(img, box))
    return output


def get_min_size(sizes):
    min_size = sizes[0]
    for size in sizes:
        min_size = np.minimum(min_size, sizes)
    return min_size[0]


def get_min_size_by_boxs(boxs):
    relative_boxs = np.array(list(map(real_box_to_relative_box, boxs)))
    return get_min_size(relative_boxs)


def img_to_size(img):
    return [0, 0, img.shape[1], img.shape[0]]


# interpolation options
# cv2.INTER_LINEAR            쌍 선형 보간법
# cv2.INTER_LINEAR_EXACT    비트 쌍 선형 보간법
# cv2.INTER_CUBIC            바이큐빅 보간법
# cv2.INTER_AREA            영역 보간법(defalt)
# cv2.INTER_LANCZOS4        Lanczos 보간법
def resize_img_all(imgs, size):
    output = []
    for img in imgs:
        output.append(cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA))
    return output


def real_box_to_relative_box(box):
    return [0, 0, box[3]-box[1], box[2]-box[0]]


def get_img_hists_by_size(imgs, size):
    return list(map(get_img_hist, resize_img_all(imgs, (size[2], size[3]))))


def putText(img, text, pivot):
    cv2.putText(img, text, pivot, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def get_middle(x1, x2):
    return int((x1 + x2) / 2)

if __name__ == '__main__':
    model, inp_dim = model_build(cfgfile, weightsfile, confidence, nms_thesh);

    cap = cv2.VideoCapture(videoURI)

    if not cap.isOpened():
        raise Exception("Could not open video device")

    print('opend: ', videoURI)
    if is_resize:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resize_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resize_height)

    # If you want to skip frames, uncomment the cap.set below
    # cap.set(1, 360)
    thickness = 3
    color = (0, 255, 0)

    prev_min_size = []
    prev_imgs = []
    img_dic_id = 0
    img_dic = {}
    late_img_dic = {}
    point_distance = 0
    if is_img_write:
        out1 = cv2.VideoWriter('run_yolo.avi', fourcc=cv2.VideoWriter_fourcc(
            *'DIVX'), fps=img_write_fps, frameSize=img_write_frame_szie)
        out2 = cv2.VideoWriter('origin.avi', fourcc=cv2.VideoWriter_fourcc(
            *'DIVX'), fps=img_write_fps, frameSize=img_write_frame_szie)
    frame_index = 0

    prevTime = 0
    try:
        while True:
            frame_index += 1

            #ret, img = cap.read()
            #if not ret:    
                #break
            ret, color_image = cap.read()
            if not ret:    
                break
            origin_color_image = color_image.copy()

            #print(frame_index)
            cut_imgs = []
            cut_min_size = []        
            yolo_img, boxs = yolo_output(color_image, model, confidence, nms_thesh, CUDA, inp_dim, is_draw)

            if len(boxs) > 0:
                cut_imgs = get_cut_imgs_by_boxs(color_image, boxs)
                cut_min_size = get_min_size_by_boxs(boxs)            

                if len(img_dic) > 0:
                    img_dic_keys = list(img_dic.keys())
                    prev_imgs = list(map(lambda img_dic_key: img_dic[img_dic_key][0],  img_dic.keys()))
                    prev_min_size = get_min_size(list(map(img_to_size, prev_imgs)))
                    late_prev_imgs = list(map(lambda img_key: late_img_dic[img_key][0],  late_img_dic.keys()))
                    late_prev_min_size = get_min_size(list(map(img_to_size, prev_imgs)))
                    min_size = get_min_size([cut_min_size, prev_min_size, late_prev_min_size])
                    cut_hists = get_img_hists_by_size(cut_imgs, min_size)
                    prev_hists = get_img_hists_by_size(prev_imgs, min_size)
                    late_prev_hists = get_img_hists_by_size(late_prev_imgs, min_size)
                    all_similarities = {}
                    cut_length = len(cut_hists)
                    prev_imgs_length = len(prev_imgs)

                    for cut_index in range(cut_length):
                        cut_hist = cut_hists[cut_index]
                        similarities = {}
                        for prev_hist_index in range(len(prev_hists)):
                            prev_hist = prev_hists[prev_hist_index]
                            ret = cv2.compareHist(cut_hist, prev_hist, HISTCMP)
                            
                            late_prev_hist = late_prev_hists[prev_hist_index]
                            ret += cv2.compareHist(cut_hist, late_prev_hist, HISTCMP)
                            ret /= 2

                            if HISTCMP_2 != None:
                                ret2 = cv2.compareHist(cut_hist, prev_hist, HISTCMP_2)
                                ret2_late_prev = cv2.compareHist(cut_hist, late_prev_hist, HISTCMP_2)
                                if HISTCMP_2 == cv2.HISTCMP_INTERSECT:
                                    hist_sum = np.sum(cut_hist)
                                    ret2 /= hist_sum
                                    ret2_late_prev /= hist_sum
                                ret2 = (ret2 + ret2_late_prev) / 2
                                ret = (ret + ret2) / 2                                    

                            ret = round(ret, 3)                
                            similarities[img_dic_keys[prev_hist_index]] = ret                        
                        similarities = sorted(similarities.items(), key=lambda x : x[1], reverse=True)                    
                        all_similarities[cut_index] = similarities
                    all_similarities = sorted(all_similarities.items(), key=lambda x : x[1][0][1], reverse=True)    

                    next_dic = img_dic.copy()
                    for img_key in img_dic:
                        next_dic[img_key][1] = (1) if type(next_dic[img_key][1]) == type([]) else next_dic[img_key][1] + 1
                        if next_dic[img_key][1] > img_drop_decision_frame and img_key != 0:
                            del next_dic[img_key]
                            del late_img_dic[img_key]
                    img_dic = next_dic                

                    #print('all_similarities')
                    #for similarities in all_similarities:
                        #print(similarities)
                    
                    if len(all_similarities) > 0:        
                        target_all_similarities_index = -1
                        target_ssenderimilarities_index = -1
                        max_hist = 0                                
                        for all_similarities_index in range(len(all_similarities)):
                            cut_key, similarities = all_similarities[all_similarities_index]
                            for similarities_index in range(len(similarities)):
                                img_key, hist = similarities[similarities_index]
                                if img_key == 0:
                                    if max_hist < hist:                                    
                                        target_all_similarities_index = all_similarities_index
                                        target_similarities_index = similarities_index
                                        max_hist = hist
                                    break

                        if target_all_similarities_index >= 0 and max_hist >= object_limitations:
                            cut_key, similarities = all_similarities[target_all_similarities_index]
                            img_key, hist = similarities[target_similarities_index]                        
                            img_dic[img_key] = [cut_imgs[cut_key], boxs[cut_key], max_hist]
                            if img_key not in late_img_dic:
                                late_img_dic[img_key] = [cut_imgs[cut_key], cut_imgs[cut_key]]
                            elif (frame_index % late_img_renewal_frame) == 0:
                                late_img_dic[img_key] = [late_img_dic[img_key][1], cut_imgs[cut_key]]


                            del all_similarities[target_all_similarities_index]
                            for similarities in all_similarities:    
                                index = -1
                                for similarity in similarities[1]:
                                    index += 1
                                    if img_key == similarity[0]:
                                        break
                                del similarities[1][index]
                            prev_imgs_length -= 1
                        

                    while len(all_similarities) > 0 and prev_imgs_length > 0:
                        cut_img_key = all_similarities[0][0]
                        (img_dic_key, first_value) = all_similarities[0][1][0]                                        
                        if first_value < object_limitations:
                            break

                        img_dic[img_dic_key] = [cut_imgs[cut_img_key], boxs[cut_img_key], first_value]
                        if img_dic_id not in late_img_dic:
                            late_img_dic[img_dic_key] = [cut_imgs[cut_img_key], cut_imgs[cut_img_key]]
                        elif (frame_index % late_img_renewal_frame) == 0:
                            late_img_dic[img_dic_key] = [late_img_dic[img_dic_key][1], cut_imgs[cut_img_key]]

                        del all_similarities[0]
                        for similarities in all_similarities:    
                            index = -1
                            for similarity in similarities[1]:
                                index += 1
                                if img_dic_key == similarity[0]:
                                    break
                            del similarities[1][index]
                        
                        prev_imgs_length -= 1        
                    
                    remain_imgs = []
                    remain_boxs = []

                    for similarities in all_similarities:
                        key = similarities[0]
                        remain_imgs.append(cut_imgs[key])                            
                        remain_boxs.append(boxs[key])

                    cut_imgs = remain_imgs
                    boxs = remain_boxs
                    
            for cut_index in range(len(cut_imgs)):
                img_dic[img_dic_id] = [cut_imgs[cut_index], boxs[cut_index], 1]
                if img_dic_id not in late_img_dic:
                    late_img_dic[img_dic_id] = [cut_imgs[cut_index], cut_imgs[cut_index]]
                elif (frame_index % late_img_renewal_frame) == 0:
                    late_img_dic[img_dic_id] = [late_img_dic[img_dic_id][1], cut_imgs[cut_index]]

                img_dic_id += 1
            
            if reset_flag:
                img_dic_id = 0
                img_dic = {}
                late_img_dic = {}
                frame_index = 0
                reset_flag_sender = 1
            else:
                reset_flag_sender = 0
    
            for key in img_dic.keys():            
                box = img_dic[key][1]
                if(type(box) == type([])):                    
                    cv2.rectangle(color_image, (box[0], box[1]), (box[2], box[3]), color, thickness)
                    center_x = (int)(get_middle(box[2], box[0]))
                    center_y = (int)(get_middle(box[1], box[3]))
                    putText(color_image, f'ID: {key}',  (center_x - 10, center_y - 50))
                    putText(color_image, f'P: ({center_x}, {center_y})',  (center_x - 10, center_y - 30))
                    putText(color_image, f'hist: {round(img_dic[key][2], 2)}',  (center_x - 10, center_y - 10))                
                    cv2.circle(color_image, (center_x, center_y), 4, (0, 255, 0), -1)

                    #print(key, img_dic[key][1], img_dic[key][2], [(img_dic[key][1][0] + img_dic[key][1][3]) / 2, (img_dic[key][1][1] + img_dic[key][1][3]) / 2 ])
                #else:                        
                    #print(key, img_dic[key][1], img_dic[key][2])
            
            #putText(color_image, f'Frame: {frame_index}',  (10, 10))
            

            

            curTime = time.time()
            sec = curTime - prevTime
            prevTime = time.time()
            #print(f"frame rate: {(1 / sec):.2f}")
            putText(color_image, f'FPS: {((int)(1 / sec))}',  (10, 20))

            if is_show:            
                cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("test", 960, 540)
                cv2.imshow("test", color_image)
                keyboard = cv2.waitKey(1)
                if keyboard & 0xFF == ord('q'):
                    break
            
            #if is_show:            
                #cv2.namedWindow("test2", cv2.WINDOW_NORMAL)
                #cv2.resizeWindow("test2", 960, 540)
                #cv2.imshow("test2", depth_colormap)
                #keyboard = cv2.waitKey(1)
                #if keyboard & 0xFF == ord('q'):
                    #break

            if is_img_write:
                out_img1 = cv2.resize(color_image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_AREA)
                out1.write(out_img1)
                if videoURI == 0:
                    out_img2 = cv2.resize(origin_color_image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_AREA)
                    out2.write(out_img2)
    except KeyboardInterrupt as e:
        exit()
    finally:
        if is_img_write:
            out1.release()
            out2.release()
        cap.release()
        cv2.destroyAllWindows()
        #print()                        

