import os
import sys
import cv2 as cv
import numpy as np
import math
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import keyboard
import time
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# openpose 패키지 경로 설정
base_path = r'C:/Users/ijs/openpose/build'
sys.path.append(fr'{base_path}/python/openpose/Release')
os.environ['PATH'] += fr';{base_path}/x64/Release;{base_path}/bin;'
SYNC_CYCLE_PER_FRAME = 30
SYNC_FRAME_STACK_SIZE = 1
# 해당 값은 SYNC_CYCLE_PER_FRAME(시퀀스 길이)를 몇 개만큼 잘라내어 비교할지를 나타내는 값
# SYNC_CYCLE_PER_FRAME을 이 값으로 나누었을 때, 나머지가 없는 양의 정수이어야 함
# 이 값이 낮아질수록 보다 좋은 싱크로율 측정 결과가 나오지만, 시간적 성능은 감소함

# openpose 패키지 import
try:
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found.')
    raise e


def two_lists_multiplication(list1, list2):
    # 길이가 같은 두 리스트의 i번째 원소들의 곱에 대한 값을 담은 리스트를 반환하는 함수
    result = []
    for i in range(len(list1)):
        result.append(list1[i] * list2[i])

    return result


def load_ref_db(db_path):  # 저장된 db(.npy 파일) 불러오기
    try:
        load_keypoints = np.load(db_path[0])
        load_frames = np.load(db_path[1])
        print('성공적으로 db를 불러왔습니다!')
        return load_keypoints, load_frames

    except FileNotFoundError:
        print('db 불러오기에 실패하였습니다. 저장된 db 파일명과 경로를 확인해주세요\n 프로그램을 종료합니다.')
        sys.exit()


def hconcat_2_videos(frame1, frame2):  # 두 영상을 수평으로 붙인 것을 반환
    hconcat_frame = cv.hconcat([frame1, frame2])

    return hconcat_frame


def hconcat_2_videos_list(frame_list1, frame_list2):
    result_frame_list = []
    for i in range(len(frame_list1)):
        hconcat_frame = cv.hconcat([frame_list1[i], frame_list2[i]])
        result_frame_list.append(hconcat_frame)

    return result_frame_list


def get_cos_two_vector(vector1, vector2):  # 두 벡터의 코사인 유사도를 반환
    cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return cos


def get_euclidean_distance(cos):  # 코사인 유사도를 이용한 유클리드 거리 측정 및 반환
    if cos > 1.0:  # numpy를 이용하여 얻은 cos 값이 가끔 1.00001 형태를 띠우므로 오류 방지를 위한 한계치 설정
        cos = 1.0
    elif cos < -1.0:
        cos = -1
    return math.sqrt(2 * (1 - cos))


def get_score(distance):  # 유클리드 거리에 반비례한 score 값 반환
    score = 100 - 100 * distance
    if score < 0:  # score가 마이너스 값이 될 수 있어서 하한치를 0으로 설정함
        score = 0

    return score


def analysis_ref_keypoints(ref_keypoints):  # ref video의 모든 Keypoint를 벡터 변환하여 반환
    ref_vector_list, ref_confidence_list = [], []

    for i in range(ref_keypoints.shape[0]):
        ref_vector_pf, ref_confidenc_pf = analysis_keypoints(ref_keypoints[i])
        ref_vector_list.append(ref_vector_pf)
        ref_confidence_list.append(ref_confidenc_pf)

    return ref_vector_list, ref_confidence_list


def analysis_keypoints(keypoints):  # 검출된 주요 부위에 대해 L2 정규화된 벡터를 구한 것과 Confidence의 평균을 반환
    essential_check_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    check_index = essential_check_index.copy()
    vector_dict, confidence_dict = dict(), dict()

    if len(keypoints.shape) == 0:  # keypoint가 하나도 검출 안되었을 때
        return 0, 0
    else:
        for index in essential_check_index:
            if np.all(keypoints[0][index] == 0):  # 검출에 필요한 요소가 하나라도 검출지 안되었는지 여부 check
                check_index.remove(index)

        if 3 in check_index and 2 in check_index:
            right_arm1 = (
                keypoints[0][3][0] - keypoints[0][2][0], keypoints[0][3][1] - keypoints[0][2][1])  # 3, 2
            right_arm1 = get_unit_vector(right_arm1)
            vector_dict['right_arm1'] = right_arm1
            confidence_dict['right_arm1'] = (keypoints[0][3][2] + keypoints[0][2][2]) / 2

        if 4 in check_index and 3 in check_index:
            right_arm2 = (
                keypoints[0][4][0] - keypoints[0][3][0], keypoints[0][4][1] - keypoints[0][3][1])  # 4, 3
            right_arm2 = get_unit_vector(right_arm2)
            vector_dict['right_arm2'] = right_arm2
            confidence_dict['right_arm2'] = (keypoints[0][4][2] + keypoints[0][3][2]) / 2

        if 6 in check_index and 5 in check_index:
            left_arm1 = (
                keypoints[0][6][0] - keypoints[0][5][0], keypoints[0][6][1] - keypoints[0][5][1])  # 6, 5
            left_arm1 = get_unit_vector(left_arm1)
            vector_dict['left_arm1'] = left_arm1
            confidence_dict['left_arm1'] = (keypoints[0][6][2] + keypoints[0][5][2]) / 2

        if 7 in check_index and 6 in check_index:
            left_arm2 = (
                keypoints[0][7][0] - keypoints[0][6][0], keypoints[0][7][1] - keypoints[0][6][1])  # 7, 6
            left_arm2 = get_unit_vector(left_arm2)
            vector_dict['left_arm2'] = left_arm2
            confidence_dict['left_arm2'] = (keypoints[0][7][2] + keypoints[0][6][2]) / 2

        if 2 in check_index and 1 in check_index:
            right_shoulder = (
                keypoints[0][2][0] - keypoints[0][1][0], keypoints[0][2][1] - keypoints[0][1][1])  # 2, 1
            right_shoulder = get_unit_vector(right_shoulder)
            vector_dict['right_shoulder'] = right_shoulder
            confidence_dict['right_shoulder'] = (keypoints[0][2][2] + keypoints[0][1][2]) / 2

        if 5 in check_index and 1 in check_index:
            left_shoulder = (
                keypoints[0][5][0] - keypoints[0][1][0], keypoints[0][5][1] - keypoints[0][1][1])  # 5, 1
            left_shoulder = get_unit_vector(left_shoulder)
            vector_dict['left_shoulder'] = left_shoulder
            confidence_dict['left_shoulder'] = (keypoints[0][5][2] + keypoints[0][1][2]) / 2

        if 1 in check_index and 0 in check_index:
            neck = (
                keypoints[0][1][0] - keypoints[0][0][0], keypoints[0][1][1] - keypoints[0][0][1])  # 1, 0
            neck = get_unit_vector(neck)
            vector_dict['neck'] = neck
            confidence_dict['neck'] = (keypoints[0][1][2] + keypoints[0][0][2]) / 2

        if 8 in check_index and 1 in check_index:
            body = (
                keypoints[0][8][0] - keypoints[0][1][0], keypoints[0][8][1] - keypoints[0][1][1])  # 8, 1
            body = get_unit_vector(body)
            vector_dict['body'] = body
            confidence_dict['body'] = (keypoints[0][8][2] + keypoints[0][1][2]) / 2

        if 9 in check_index and 8 in check_index:
            right_pelvis = (
                keypoints[0][9][0] - keypoints[0][8][0], keypoints[0][9][1] - keypoints[0][8][1])  # 9, 8
            right_pelvis = get_unit_vector(right_pelvis)
            vector_dict['right_pelvis'] = right_pelvis
            confidence_dict['right_pelvis'] = (keypoints[0][9][2] + keypoints[0][8][2]) / 2

        if 12 in check_index and 8 in check_index:
            left_pelvis = (
                keypoints[0][12][0] - keypoints[0][8][0], keypoints[0][12][1] - keypoints[0][8][1])  # 12, 8
            left_pelvis = get_unit_vector(left_pelvis)
            vector_dict['left_pelvis'] = left_pelvis
            confidence_dict['left_pelvis'] = (keypoints[0][12][2] + keypoints[0][8][2]) / 2

        if 10 in check_index and 9 in check_index:
            right_leg1 = (
                keypoints[0][10][0] - keypoints[0][9][0], keypoints[0][10][1] - keypoints[0][9][1])  # 10, 9
            right_leg1 = get_unit_vector(right_leg1)
            vector_dict['right_leg1'] = right_leg1
            confidence_dict['right_leg1'] = (keypoints[0][10][2] + keypoints[0][9][2]) / 2

        if 11 in check_index and 10 in check_index:
            right_leg2 = (
                keypoints[0][11][0] - keypoints[0][10][0], keypoints[0][11][1] - keypoints[0][10][1])  # 11, 10
            right_leg2 = get_unit_vector(right_leg2)
            vector_dict['right_leg2'] = right_leg2
            confidence_dict['right_leg2'] = (keypoints[0][11][2] + keypoints[0][10][2]) / 2

        if 13 in check_index and 12 in check_index:
            left_leg1 = (
                keypoints[0][13][0] - keypoints[0][12][0], keypoints[0][13][1] - keypoints[0][12][1])  # 13, 12
            left_leg1 = get_unit_vector(left_leg1)
            vector_dict['left_leg1'] = left_leg1
            confidence_dict['left_leg1'] = (keypoints[0][13][2] + keypoints[0][12][2]) / 2

        if 14 in check_index and 13 in check_index:
            left_leg2 = (
                keypoints[0][14][0] - keypoints[0][13][0], keypoints[0][14][1] - keypoints[0][13][1])  # 14, 13
            left_leg2 = get_unit_vector(left_leg2)
            vector_dict['left_leg2'] = left_leg2
            confidence_dict['left_leg2'] = (keypoints[0][14][2] + keypoints[0][13][2]) / 2

        return vector_dict, confidence_dict


def get_unit_vector(vector):  # 벡터의 L2 정규화 함수 (= 단위벡터를 구하는 함수)
    a = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    result = (vector[0] / a, vector[1] / a)

    return result


def scoring_two_vector(current_frame, ref_vectors, rendered_vectors, op_confidences, ref_confidences):  # scoring 함수
    scoring_points = list(set(list(ref_vectors.keys())) & set(list(rendered_vectors.keys())))
    # ref와 cam의 공통적으로 검출된 keypoint list
    sum_score, sum_confidence = 0, 0

    for point in scoring_points:
        cos = get_cos_two_vector(ref_vectors[point], rendered_vectors[point])
        distance = get_euclidean_distance(cos)
        score = get_score(distance)
        sum_score += score
        sum_confidence += (op_confidences[point] + ref_confidences[point])

    #avg_score1 = round(sum_score / len(ref_vectors), 2)  # 현 frame의 최종 score
    avg_score = round(sum_score / len(scoring_points), 2)  # 현 frame의 최종 score
    confidence = round((sum_confidence / 28) * 100, 2)  # 현 frame의 최종 confidence

    print(f"Frame {current_frame}'s score : {avg_score} % / confidence score : "
          f"{confidence} %")

    return avg_score, confidence


def print_result_text(frame_num, score_list, confidence_list, fps_avg, num_sync):  # 동작 비교 결과를 텍스트로 출력하는 함수
    print('-----------------------------------------------')
    print(f'Total Frame Count: {frame_num}')
    print(f'Average Score: {round(sum(score_list) / len(score_list), 2)} % ')
    print(f'Average Confidence: {round(sum(confidence_list) / len(confidence_list), 2)} % ')

    if num_sync * SYNC_CYCLE_PER_FRAME > frame_num / 2:
        print("Cam camera's pose is overall slow...")
    elif num_sync * SYNC_CYCLE_PER_FRAME > frame_num / 4:
        print("Cam camera's pose is partially slow...")

    print(f"FPS Average : {fps_avg}")
    print('------------------------------------------------')


def draw_graph_1_data(frame_num_list, data):  # 프레임별 FPS를 그래프로 출력해주는 함수
    plt.plot(frame_num_list, data, label='FPS')
    plt.xlabel('Frame Numbers')
    plt.ylabel('FPS value')
    plt.title("FPS's degree of change about frame")
    plt.legend()
    plt.show()


def draw_graph_result(frame_num_list, score_list, confidence_list, sync_data):  # 모든 결과에 대해 그래프 출력
    plt.plot(frame_num_list, score_list, label='Score')
    plt.plot(frame_num_list, confidence_list, label='Confidence')

    slow_label = 'Slow sync'

    if len(sync_data) > 0:
        for i in range(len(sync_data)):
            plt.axvspan(sync_data[i][0][0], sync_data[i][0][1], label=slow_label, color='cornflowerblue',
                        alpha=(sync_data[i][1]/SYNC_CYCLE_PER_FRAME))
            slow_label = ""

    plt.xlabel('Frame Numbers')
    plt.ylabel('Degree')

    plt.title('Human Pose Comparing Result')
    plt.legend()
    plt.show()


# def draw_graph_realtime(current_frame_num_list, current_score_list, current_confidence_list):
#     # 실시간적
#     plt.xlabel('Frame Numbers')
#     plt.ylabel('Degree')
#     plt.title('Human Pose Comparing real time result')
#
#     plt.plot(current_frame_num_list, current_score_list, label='Score')
#     plt.plot(current_frame_num_list, current_confidence_list, list='Confidence')
#     plt.pause(0.1)
#
#     plt.show()


def print_best_frame_by_unit(frame_num, score_list, confidence_list, result_frame_list):
    # 구간별 최고 점수를 가지는 프레임에 대한 결과를 출력하는 함수
    print('----------------------------------------------------')
    if frame_num < 8:
        print('Frame numbers less than 8....')
        print('We need at least 8 frames to print best frame result!')
        return 0

    frame_unit = int(frame_num / 8)
    current_frame_unit, previous_frame_unit = 0, 0
    score_confidence_list = two_lists_multiplication(score_list, confidence_list)
    best_index_list = []

    for i in range(8):
        current_frame_unit += frame_unit
        try:
            if i != 7:
                best_score = max(score_confidence_list[previous_frame_unit:current_frame_unit])
                best_score_index = score_confidence_list[previous_frame_unit:current_frame_unit].index(best_score)
            else:
                best_score = max(score_confidence_list[previous_frame_unit:])
                best_score_index = score_confidence_list[previous_frame_unit:].index(best_score)
        except ValueError as e:
            print('Score & Confidence list is empty!!!')

        best_score_index += previous_frame_unit
        best_index_list.append(best_score_index)
        previous_frame_unit = current_frame_unit

    current_count = 0
    for j in range(2):
        fig = plt.figure()
        fig.canvas.set_window_title(f"Best Score's Frames")

        for k in range(4):
            plt.subplot(2, 2, k + 1)
            frame = result_frame_list[best_index_list[current_count]][:, :, ::-1]  # BGR -> RGB 변환
            plt.imshow(frame)

            if current_count != 7:
                plt.title(f"[{current_count * frame_unit} ~ {(current_count + 1) * frame_unit}]  Best Frame : "
                          f"{best_index_list[current_count]} / Score : "
                          f"{round(score_list[best_index_list[current_count]], 2)} / Confidence : "
                          f"{round(confidence_list[best_index_list[current_count]], 2)}")

            else:
                plt.title(f"[{current_count * frame_unit} ~ {frame_num}]  Best Frame : "
                          f"{best_index_list[current_count]} / Score : "
                          f"{round(score_list[best_index_list[current_count]], 2)} / Confidence : "
                          f"{round(confidence_list[best_index_list[current_count]], 2)}")
            plt.axis('off')
            current_count += 1

        plt.show()


def print_best_part(score_list, confidence_list, result_frame_list, fps_list):
    # 전체 구간 중 점수가 가장 높은 프레임 +- 30 만큼 동영상으로 재생하는 함수
    print('------------------------------------------------------')
    if len(result_frame_list) <= 61:
        print('Frame numbers equal or less than 61....')
        print('We need at least 62 frames to print best part!')
        return 0

    score_confidence_list = two_lists_multiplication(score_list, confidence_list)
    best_score = max(score_confidence_list)
    best_score_index = score_confidence_list.index(best_score)
    best_part = [best_score_index, best_score_index]

    if best_score_index - 30 < 0:
        best_part[0] = 0
    else:
        best_part[0] -= 30

    if best_score_index + 30 > len(result_frame_list):
        best_part[1] = len(result_frame_list) - 1
    else:
        best_part[1] += 30

    best_part_score_avg = round(sum(score_list[best_part[0]:best_part[1]]) / (best_part[1] - best_part[0]), 2)
    best_part_confidece_avg = round(sum(confidence_list[best_part[0]:best_part[1]]) / (best_part[1] - best_part[0]), 2)
    best_part_fps_avg = round(sum(fps_list[best_part[0]:best_part[1]]) / len(fps_list[best_part[0]:best_part[1]]), 2)

    print(f"Best Part : {best_part[0]} ~ {best_part[1]}")
    print(f"Best Part's Average Score : {best_part_score_avg} % ")
    print(f"Best Part's Confidence Score : {best_part_confidece_avg} % ")
    print(f"Best Part's Average FPS : {best_part_fps_avg}")
    for i in range(best_part[1] - best_part[0]):
        cv.imshow(f"Best Part : {best_part[0]} ~ {best_part[1]}", result_frame_list[best_part[0] + i])
        if cv.waitKey(int(1000 / best_part_fps_avg)) & 0xff == 27:  # waitKey의 인자는 int형이어야함
            break


def get_dtw_distance(a, b):
    # dtw distance를 구해서 반환하는 함수
    distance, path = fastdtw(a, b, dist=euclidean)
    return distance


def analysis_sync(current_frame, ref_vector_stack, cam_vector_stack, fps_list, sync_cycle):
    # 싱크로율을 분석하여 결과 출력 및 반환하는 함수
    average_dist = 0.0
    for i in range(sync_cycle):
        try:
            if int(len(ref_vector_stack[i].keys())) == 14 and int(len(cam_vector_stack[i].keys())) == 14:
                # 주요 벡터가 다 검출이 되었다면
                ref_vector_list = list(ref_vector_stack[i].values())
                cam_vector_list = list(cam_vector_stack[i].values())
                temp, count = 0, 0
                for j in range(14):
                    dist = get_dtw_distance(ref_vector_list[j], cam_vector_list[j])
                    temp += dist
                    if dist <= 0.3:
                        count += 1
                    # print(f"{current_frame - 29 + i} : {list(ref_vector_stack[i].keys())[j]}'s dtw "
                    #       f"distance is {dist}")
                # print(f"Average dtw score : {temp / 14} / 0.3 under count : {count}/14")
                if count < 10:
                    return 0
                average_dist += temp

            else:
                # print(f'Any essential keypoint is not detected...')
                return 0
        except AttributeError:  # 벡터 검출이 안된 프레임이 있다면 함수 종료
            return 0
    average_dist /= (sync_cycle * 14)
    # print(f"{current_frame - 30} ~ {current_frame}'s total average dtw distance : {average_dist}")

    if 0.45 > average_dist > 0.15:  # 해당 sequence의 평균 dtw distance가 0.15 ~ 0.45 범위에 있을 경우
        ref_stack_flatten, cam_stack_flatten, dtw_dist_list = [], [], []

        for i in range(14):
            for j in range(sync_cycle):
                ref_stack_flatten.append(list(ref_vector_stack[j].values())[i])
                cam_stack_flatten.append(list(cam_vector_stack[j].values())[i])

        for k in range(int(SYNC_CYCLE_PER_FRAME / SYNC_FRAME_STACK_SIZE)):
            temp_dist = 0
            for l in range(14):
                temp_dist += get_dtw_distance(ref_stack_flatten[l * sync_cycle:(l + 1) * sync_cycle],
                                              cam_stack_flatten[(k * SYNC_FRAME_STACK_SIZE + l * sync_cycle):(l + 1)
                                                                                                             * sync_cycle])
            dtw_dist_list.append(temp_dist / 14)
        dtw_min = min(dtw_dist_list)
        fps_avg = sum(fps_list) / len(fps_list)

        best_dist_index = dtw_dist_list.index(dtw_min)
        print(f'Frame {current_frame - sync_cycle + 1} ~ {current_frame} 구간의 피험자 동작이 약 '
              f'{round((SYNC_FRAME_STACK_SIZE * (best_dist_index + 1)) / fps_avg, 2)}초 '
              f'({SYNC_FRAME_STACK_SIZE * (best_dist_index + 1)} 프레임) 만큼 느립니다.')
        return [[current_frame - sync_cycle, current_frame], SYNC_FRAME_STACK_SIZE * (best_dist_index + 1)]

    else:
        # print('Cannot find sync similiarity...')
        return 0


def sync_data_filtering(sync_data_list):
    result = []
    for sync_data in sync_data_list:
        if sync_data == 0:
            continue
        else:
            result.append(sync_data)

    return result


def sync_result_analysis(ref_frame_list, ref_vector_list, cam_frame_list, cam_vector_list, sync_data, fps_list):
    sync_count = len(sync_data)
    for i in range(sync_count):
        frame_number_list = []
        print('---------------------------------------------------------------------------------------------------'
              '--------')
        print(f'{sync_count}개의 싱크로율 문제 부분 중 {i + 1}번째 부분의 분석입니다. \n* 종료하시려면 ESC 버튼을 누르세요')

        # if keyboard.is_pressed('Esc'):  # Esc 버튼을 누르면 종료
        #     print('ESC 버튼을 눌러 싱크로율 분석을 종료합니다.')
        #     return 0

        print(f'{i + 1}번째 구간의 원본 영상입니다.')
        original = hconcat_2_videos_list(cam_frame_list[sync_data[i][0][0]:sync_data[i][0][1]],
                                         ref_frame_list[sync_data[i][0][0]:sync_data[i][0][1]])
        for j in range(SYNC_CYCLE_PER_FRAME):
            cv.imshow('Original Video', original[j])
            if cv.waitKey(int(1000 / fps_list[sync_data[i][0][0] + j])) & 0xff == 27:
                cv.destroyAllWindows()
                return 0
            frame_number_list.append(sync_data[i][0][0] + j)

        time.sleep(2)

        print(f'{i + 1}번째 구간의 싱크로율 조정 후 영상입니다.')

        if sync_data[i][0][1] + sync_data[i][1] >= len(cam_frame_list) - 1:
            print('해당 구간은 캠카메라의 가장 마지막 부분이므로 싱크로율 조정을 할 수 없습니다...')
        else:
            fix_sync = hconcat_2_videos_list(cam_frame_list[sync_data[i][0][0] + sync_data[i][1]:
                                                            sync_data[i][0][1] + sync_data[i][1]],
                                             ref_frame_list[sync_data[i][0][0]:sync_data[i][0][1]])
            for k in range(SYNC_CYCLE_PER_FRAME):
                cv.imshow('Original Video', fix_sync[k])
                if cv.waitKey(int(1000 / fps_list[sync_data[i][0][0] + k])) & 0xff == 27:
                    cv.destroyAllWindows()
                    return 0

            origin_score_list, fix_score_list = [], []

            for l in range(SYNC_CYCLE_PER_FRAME):
                origin_sum_score, fix_sum_score = 0, 0
                scoring_points = list(set(list(ref_vector_list[sync_data[i][0][0] + l].keys())) &
                                      set(list(cam_vector_list[sync_data[i][0][0] + l].keys())))
                for point in scoring_points:
                    origin_cos = get_cos_two_vector(ref_vector_list[sync_data[i][0][0] + l][point],
                                                    cam_vector_list[sync_data[i][0][0] + l][point])
                    origin_distance = get_euclidean_distance(origin_cos)
                    origin_score = get_score(origin_distance)
                    origin_sum_score += origin_score

                    fix_cos = get_cos_two_vector(ref_vector_list[sync_data[i][0][0] + l][point],
                                                 cam_vector_list[sync_data[i][0][0] + l + sync_data[i][1]][point])
                    fix_distance = get_euclidean_distance(fix_cos)
                    fix_score = get_score(fix_distance)
                    fix_sum_score += fix_score

                origin_score_list.append(origin_sum_score/14)
                fix_score_list.append(fix_sum_score/14)

            origin_avg_score = round(sum(origin_score_list) / SYNC_CYCLE_PER_FRAME, 2)
            fix_avg_score = round(sum(fix_score_list) / SYNC_CYCLE_PER_FRAME, 2)

            print(f'원본의 {i + 1}번째 구간({sync_data[i][0][0]} ~ {sync_data[i][0][1]} 프레임)의 평균 Score : {origin_avg_score}%')
            print(f'개선본의 {i + 1}번째 구간({sync_data[i][0][0]} ~ {sync_data[i][0][1]} 프레임)의 평균 Score : {fix_avg_score}%')
            print(f'개선 후 약 {round((fix_avg_score - origin_avg_score)/origin_avg_score, 3) * 100}% 만큼 Score가 향상되었습니다!')

            plt.plot(frame_number_list, origin_score_list, label="Original Score")
            plt.plot(frame_number_list, fix_score_list, label="Fixed Score")
            plt.xlabel('Frame Numbers')
            plt.ylabel('Score')

            plt.title(f"{sync_data[i][0][0]} ~ {sync_data[i][0][1]} frames'Score graph")
            plt.legend()
            plt.show()

        time.sleep(2)
    print('-------------------------------------------------------------------------------------------------')


# def get_weighted_confidence_score(scoring_points, op_confidences, unit_vector1, unit_vector2):
#     confidence_sum = 0
#     temp = 0
#     for point in scoring_points:
#         confidence_sum += op_confidences[point]
#     for point2 in scoring_points:
#         temp += op_confidences[point2] * np.linalg.norm(unit_vector2 - unit_vector1)


def main():
    ref_db_path = []
    ref_keypoints_path = 'ref_video_2_short2_keypoints.npy'
    ref_db_path.append(ref_keypoints_path)

    print('ref 동영상의 Blending 여부를 설정해주세요...')
    print('* blending on : 실제 모습이 가려짐  /  blending off : 실제 모습이 드러남')
    print('1. Blending on       2. Blending off')
    ref_blending_check_input = input('입력 값 : ')

    if ref_blending_check_input == '1':
        frames_path = 'ref_video_2_short2_frames_blending_True.npy'
        ref_db_path.append(frames_path)
        print('ref video 가 Blending On 으로 설정되었습니다...')
    else:
        frames_path = 'ref_video_2_short2_frames_blending_False.npy'
        ref_db_path.append(frames_path)
        print('ref video 가 Blending Off 으로 설정되었습니다...')
    print('-------------------------------------------------------------------------')

    ref_keypoints, ref_rendered_frames = load_ref_db(ref_db_path)
    ref_vectors, ref_confidences = analysis_ref_keypoints(ref_keypoints)
    # print(ref_rendered_frames.shape)

    params = {
        'model_folder': fr'{base_path}/../models/',
        'model_pose': 'BODY_25',
        'number_people_max': 1,
        'num_gpu_start': 0,
        'disable_blending': False  # blending 유무.
    }

    print('Cam camera의 Blending 여부를 설정해주세요...')
    print('* blending on : 실제 모습이 가려짐  /  blending off : 실제 모습이 드러남')
    print('1. Blending on       2. Blending off')
    blending_check_input = input('입력 값 : ')

    if blending_check_input == '1':
        params['disable_blending'] = False
        print('Cam camera 가 Blending On 으로 설정되었습니다...')
    else:
        params['disable_blending'] = True
        print('Cam camera 가 Blending Off 으로 설정되었습니다...')
    print('-------------------------------------------------------------------------')

    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()
    datum = op.Datum()

    cap = cv.VideoCapture(0)
    assert cap.isOpened(), 'Failed to initialize video capture!'
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = 'CamCamera & Redendered reference Video'
    cv.namedWindow(window_name)
    score_list, confidence_list, frame_num_list, result_frame_list, cam_vector_stack, sync_data_list, fps_list \
        = [], [], [], [], [], [], []
    cam_frame_list, cam_vector_list = [], []
    prev_time, rendering_win_name, real_time_graph_name = 0, 'Rendered Cam Video & Rendered Referece Video', ''

    real_time_graph = plt.figure()  # 실시간 그래프 크기 조절
    real_time_graph_mng = plt.get_current_fig_manager()
    real_time_graph_mng.window.setGeometry(0, 540, 1280, 480)

    plt.xlabel('Frame Numbers')
    plt.ylabel('Degree')
    plt.title('Human Pose Comparing real time result')

    prev_time = 0
    for i in range(len(ref_rendered_frames)):
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab a frame!')
            break

        frame = cv.flip(frame, 1)
        datum.cvInputData = frame
        op_wrapper.emplaceAndPop([datum])
        keypoints = datum.poseKeypoints
        rendered_vector, op_confidences = analysis_keypoints(keypoints)
        cam_vector_stack.append(rendered_vector)

        current_time = time.time()
        sec = current_time - prev_time
        prev_time = current_time
        current_fps = 1 / (sec)
        fps_list.append(current_fps)
        # print(f'Current FPS : {current_fps}')

        if rendered_vector == 0 or ref_vectors[i] == 0:
            print(f'Cannot detect any keypoints in reference video or cam video.....')
            score_list.append(0)
            confidence_list.append(0)
        else:
            score, confidence = scoring_two_vector(i, ref_vectors[i], rendered_vector, op_confidences,
                                                   ref_confidences[i])
            score_list.append(score)
            confidence_list.append(confidence)

        if i != 0 and i % SYNC_CYCLE_PER_FRAME == 0:  # SYNC_CYCLE 값의 프레임 단위로
            sync_check_score = sum(score_list[i - SYNC_CYCLE_PER_FRAME:i]) / SYNC_CYCLE_PER_FRAME
            # 최근 SYNC_CYCLE 범위의 프레임에 대한 평균 SCORE 계산
            if sync_check_score < 85:  # 해당 범위의 평균 Score가 85 미만이면
                sync_data = analysis_sync(i, ref_vectors[i - SYNC_CYCLE_PER_FRAME:i],
                                          cam_vector_stack[i - SYNC_CYCLE_PER_FRAME:i],
                                          fps_list[i - SYNC_CYCLE_PER_FRAME:i], SYNC_CYCLE_PER_FRAME)
                sync_data_list.append(sync_data)

        frame_num_list.append(i)
        rendered_frame = datum.cvOutputData
        cam_frame_list.append(rendered_frame)
        result_frame = hconcat_2_videos(rendered_frame, ref_rendered_frames[i])
        result_frame_list.append(result_frame)

        cv.namedWindow(rendering_win_name)
        cv.moveWindow(rendering_win_name, 0, 0)
        cv.imshow(rendering_win_name, result_frame)

        plt.plot(frame_num_list, score_list, color='red', label='Score')
        plt.plot(frame_num_list, confidence_list, color='blue', label='Confidence')
        # plt.xticks(range(0, len(frame_num_list)+1, step=))

        if i == 0:
            plt.legend()
        plt.pause(0.0001)

        if i == len(ref_rendered_frames) - 1:
            print("Reference Video's playing is done!!!")
            cv.destroyAllWindows()
            plt.close(real_time_graph)
            break

        if keyboard.is_pressed('Esc'):  # Esc 버튼을 누르면 종료
            plt.close(real_time_graph)
            print('Esc is pressed, so real time graph is now closed...')
            break

        if cv.waitKey(33) & 0xff == 27 or i == len(ref_rendered_frames) - 1:  # waitKey의 인자는 int형이어야함
            break

    plt.show()
    cap.release()
    cv.destroyAllWindows()
    fps_avg = sum(fps_list) / len(fps_list)
    sync_data = sync_data_filtering(sync_data_list)
    print_result_text(len(result_frame_list), score_list, confidence_list, fps_avg, len(sync_data))
    draw_graph_result(frame_num_list, score_list, confidence_list, sync_data)

    if len(sync_data) >= 1:
        print(f'싱크가 맞지 않는 부분이 {len(sync_data)}개 있습니다.')
        slow_anal_check = input(f'싱크로율 분석 & 개선 과정을 보시겠습니까? (예 : 1, 아니오 : 0 입력...) : ')
        if slow_anal_check == '1':
            sync_result_analysis(ref_rendered_frames, ref_vectors, cam_frame_list, cam_vector_stack, sync_data,
                                 fps_list)
        else:
            pass

    draw_graph_1_data(frame_num_list, fps_list)
    _ = print_best_frame_by_unit(len(result_frame_list), score_list, confidence_list, result_frame_list)
    _ = print_best_part(score_list, confidence_list, result_frame_list, fps_list)


if __name__ == '__main__':
    main()
