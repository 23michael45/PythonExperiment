import cv2 
import numpy as np
import sys




filePath = "LibPano/Images/Calibrate/Fisheye/"
caliberationResultFileName= "caliberation_result.txt" #保存定标结果的文件


#读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化  

image_count=  14;						#图像数量
board_size = (9,6);             #定标板上每行、列的角点数

x_expand = 0							#x,y方向的扩展(x横向，y纵向)，适当增大可以不损失原图像信息
y_expand = 200;			

corners =  None 			#缓存每幅图像上检测到的角点
corners_Seq = []  				#保存检测到的所有角点/   
image_Seq = []
successImageNum = 0;				#成功提取角点的棋盘图数量	
conner_flag = True;				#所有图像角点提取成功为true，其余为false

for i in np.arange(1,image_count + 1):
	fileName = '{0}{1}{2}.jpg'.format(filePath,'img',i)
	srcimg = cv2.imread(fileName,cv2.IMREAD_COLOR)

	image = None								#边界扩展后的图片
	image = cv2.copyMakeBorder(srcimg,np.int(y_expand/2),np.int(y_expand/2),np.int(x_expand/2),np.int(x_expand/2),cv2.BORDER_CONSTANT)
	
	#提取角点
	
	imageGray = None
	imageGray = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY);
	patternfound ,corners= cv2.findChessboardCorners(image, board_size,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK );
	if not patternfound:   
		cout<<"img{0}{1}".format({i},"角点提取失败，请删除该图片，重新排序后再次运行程序！")
		conner_flag = false;
		break;
	else:
		#亚像素精确化
		stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
		cv2.cornerSubPix(imageGray,corners,(11,11),(-1,-1),stop_criteria)
		#绘制检测到的角点并保存
		imageTemp = image;
		for j in np.arange(0,corners.shape[0]):

			center = tuple(corners[j,0,:])
			cv2.circle(imageTemp,center,10,(0,0,255),2,8,0)

		
		saveFileName = '{0}{1}{2}.jpg'.format(filePath,'corner',i)
		print("保存文件{0}".format(saveFileName))
		cv2.imwrite(saveFileName,imageTemp)


		corners_Seq.append(corners)
		successImageNum += 1
	image_Seq.append(image)

if (not conner_flag): #如果有提取失败的标定图，退出程序
	sys.exit(0)

print("角点提取完成！")



# 摄像机定标  


print("开始定标………………！")

square_size = (20,20)

object_Points = [];        #保存定标板上角点的三维坐标
point_counts = []


CHECKERBOARD = (9, 7)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


for t in np.arange(0,successImageNum):
	'''
	tempPointSet = np.zeros((1,board_size[0]*board_size[1],3))
	for i in np.arange(0,board_size[1]):#height
		for j in np.arange(0,board_size[0]):#width
			#假设定标板放在世界坐标系中z=0的平面上
			tempPointSet[0,i * board_size[1] + j ,0] = i*square_size[0];
			tempPointSet[0,i * board_size[1] + j ,1] = j*square_size[1];
			tempPointSet[0,i * board_size[1] + j ,2] = 0;
	'''
	
	tempPointSet = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
	tempPointSet[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size[0]
	object_Points.append(tempPointSet)

for i in np.arange(0,successImageNum):
	point_counts.append(board_size[0] * board_size[1])

image_size = image_Seq[0].shape[0:2]
intrinsic_matrix = np.zeros((3,3))#摄像机内参数矩阵
distortion_coeffs = np.zeros((4))#摄像机的4个畸变系数：k1,k2,k3,k4
rotation_vectors = []#每幅图像的旋转向量
translation_vectors = []#每幅图像的平移向量
flags = 0;
flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC;
#flags |= cv2.fisheye.CALIB_CHECK_COND;
flags |= cv2.fisheye.CALIB_FIX_SKEW;


stop_criteria = (3, 20, 1e-6)
rms,intrinsic_matrix,distortion_coeffs,rotation_vectors,translation_vectors =   cv2.fisheye.calibrate(object_Points, corners_Seq,image_size,None, None, None, None, flags,stop_criteria);
print("定标完成！")


# 对定标结果进行评价  

print("开始评价定标结果………………")
total_err = 0.0;#所有图像的平均误差的总和
err = 0.0; #每幅图像的平均误差
image_points2 = []#保存重新计算得到的投影点


for i in np.arange(0,successImageNum):
	tempPointSet = object_Points[i];
	#通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点
	image_points2,jac  = cv2.fisheye.projectPoints(tempPointSet,rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs)
	#计算新的投影点和旧的投影点之间的误差
	tempImagePoint = corners_Seq[i]

	tempImagePointMat = []
	image_points2Mat = []
	for t in np.arange(0,tempImagePoint.shape[0]):
		image_points2Mat.append(image_points2[0,t,:])
		tempImagePointMat.append(tempImagePoint[t].reshape(2))

	image_points2Mat = np.array(image_points2Mat)
	tempImagePointMat = np.array(tempImagePointMat)
	
	err = cv2.norm(image_points2Mat-tempImagePointMat,cv2.NORM_L2)
	err/=  point_counts[i];   
	total_err += err
	print("第{0}幅图像的平均误差：{1}像素".format(i+1,err))

print("总体平均误差:{0}像素".format(total_err))



#显示定标结果
print("保存矫正图像")
for i in np.arange(0,successImageNum):
	R = np.eye(3)
	
	mapx,mapy = cv2.fisheye.initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,image_size,cv2.CV_32FC1)
	t = image_Seq[i]
	t = cv2.remap(t,mapx,mapy,cv2.INTER_LINEAR)

	saveFileName = '{0}{1}{2}.jpg'.format(filePath,'result',i+1)
	print("保存文件{0}".format(saveFileName))
	cv2.imwrite(saveFileName,t)