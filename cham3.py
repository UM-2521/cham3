"""
우울증 환자의 증상 완화를 위해 게임을 같이 하는 Q.bo 로봇
Q.bo 로봇이 참참참 게임 인터랙션을 할 때 머무를 코드입니다.

참참참 게임에서 큐보가 공격 입장이 되고 사람이 고개를 돌리는 입장입니다.
[1] '참, 참, 참'을 큐보가 외칩니다.
[2] 3을 외치는 순간 큐보는 왼쪽/오른쪽 중 1개를 말하고, 
    사람은 동시에 왼쪽/오른쪽 중 한쪽으로 고개를 돌립니다.
[3] 두 방향이 일치하면 큐보가 이겼다/ 불일치하면 졌다고 말해줍니다.
"""

import random
import time

import cv2 as cv
import numpy as np

# Q.bo의 눈(카메라)를 사용
cap = cv.VideoCapture(0)

# 피부색의 범위 설정
lower = np.array([20, 70, 120], dtype="uint8")
upper = np.array([160, 225, 255], dtype="uint8")

# 얼굴 인식을 위한 학습 모델 가져오기
folder = 'C:/Users/ekrmd/Anaconda3/envs/bluesea/Lib/site-packages/cv2/data/'
face_cascade = cv.CascadeClassifier(folder + 'haarcascade_frontalface_default.xml')


def run():
    """
    Q.bo 로봇이 사람과 참참참 게임을 하는 코드입니다.
    
    opencv로 매 프레임을 가져와 얼굴 인식을 시도합니다.
    얼굴 인식에 성공하면 참참참을 외칩니다.
    (음성 파일은 print문으로 대체하였습니다.)

    그후 큐보는 랜덤하게 방향 한쪽을 고르고 유저의 고개 방향 분석합니다.
    참참참 외치기 전 얼굴 이미지의 왼쪽 절반과
    외친 후 얼굴 이미지 왼쪽 절반을 가져온 다음
    위에서 정의한 피부색 범위 안에 있는 픽셀이 더 많은 쪽을 평균을 이용해 구합니다.
    """
    while True:
        ret, frame = cap.read()

        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for x, y, w, h in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for i in range(3):
                    print('cham')
                    time.sleep(0.5)

                qbo_choice = random.randrange(0, 2)
                if qbo_choice == 0:
                    print('Q.bo : Left')
                else:
                    print('Q.bo : Right')

                # 1.5초 만큼 프레임이 밀렸으니 45 frame 만큼 넘어감
                for i in range(45):
                    ret2, new_frame = cap.read()

                face_left_before = cv.inRange(frame[y:y + h, x:x + int(w / 2)], lower, upper)
                face_left_after = cv.inRange(new_frame[y:y + h, x:x + int(w / 2)], lower, upper)

                if np.mean(face_left_before) < np.mean(face_left_after):
                    # Q.bo 입장에서 상대 얼굴이 왼쪽으로 돌아감
                    print('User : Left')
                    user_choice = 0
                else:
                    print('User : Right')
                    user_choice = 1

                if qbo_choice == user_choice:
                    print("Q.bo's attck success!")
                else:
                    print("Q.bo's attck failure...")

            cv.imshow('frame', frame)

        k = cv.waitKey(5) & 0xFF
        if k == 27:  # ESC Key
            break

    cap.release()
    cv.destroyAllWindows()
