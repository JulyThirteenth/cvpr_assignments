import numpy as np
import cv2
import time
import datetime

cap = cv2.VideoCapture("GOPR0110.MP4")#打开一个视频

fourcc = cv2.VideoWriter_fourcc(*'XVID')#设置保存图片格式
out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 10.0, (1920,1080))#分辨率要和原视频对应

CornerPoint = 'Harris' # ShiTomasi or Harris
if CornerPoint == 'ShiTomasi':
    # ShiTomasi 角点检测参数
    feature_params = dict(maxCorners = 300,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7)
else:
    feature_params = dict(maxCorners = 300,
                        qualityLevel = 0.1,
                        minDistance = 7,
                        blockSize = 7,
                        useHarrisDetector = True,
                        k = 0.04)

# lucas kanade光流法参数
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建随机颜色
color = np.random.randint(0,255,(100,3))

# 获取第一帧，找到角点
ret, old_frame = cap.read()
#找到原始灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#获取图像中的角点，返回到p0中
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# 创建一个蒙版用来画轨迹
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read() #读取图像帧
    if frame is not None:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #灰度化

        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        while p1 is None:
            print("Optical flow track failed!")
            # 重新获取图像中的角点，返回到p0中
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            # 重新计算光流
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # 选取好的跟踪点
        good_new = p1[st==1]
        good_old = p0[st==1]

        # 画出轨迹
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()#多维数据转一维,将坐标转换后赋值给a，b
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)#画直线
            frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)#画点
        img = cv2.add(frame,mask) # 将画出的线条进行图像叠加

        cv2.imshow('frame',img)  #显示图像

        out.write(img)#保存每一帧画面

        # 更新上一帧的图像和追踪点
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    
    k = cv2.waitKey(30) & 0xff #按Esc退出检测
    if k == 27:
        break

out.release()#释放文件
cap.release()
cv2.destroyAllWindows()#关闭所有窗口
