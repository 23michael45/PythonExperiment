import cv2 
import numpy as np
import sys
import ImageTransformation3D

def pyinitUndistortRectifyMap( K,D,R,T,P,size):

    map1 = np.zeros((size[1],size[0]))
    map2 = np.zeros((size[1],size[0]))
    
    #从内参矩阵K中取出归一化焦距fx,fy; cx,cy
    f = np.array([K[0, 0], K[1, 1]]);
    c = np.array([K[0, 2], K[1, 2]]);
    
    #从畸变系数矩阵D中取出畸变系数k1,k2,k3,k4
    k = D[0:4]

    
    #旋转矩阵RR转换数据类型为CV_64F，如果不需要旋转，则RR为单位阵
    RR  = np.eye(3)
    RR,jacb = cv2.Rodrigues(R)
    
    
    #新的内参矩阵PP转换数据类型为CV_64F
    PP  = np.eye(3)
    PP = P

    #关键一步：新的内参矩阵*旋转矩阵，然后利用SVD分解求出逆矩阵iR，后面用到
    iR = np.linalg.pinv(np.dot(PP , RR))

    #反向映射，遍历目标图像所有像素位置，找到畸变图像中对应位置坐标(u,v)，并分别保存坐标(u,v)到mapx和mapy中
    for i in np.arange(size[1]):#height
    
        print(i)
        #二维图像平面坐标系->摄像机坐标系
        #_x = i*iR[0, 1] + iR[0, 2],
        #_y = i*iR[1, 1] + iR[1, 2],
        #_w = i*iR[2, 1] + iR[2, 2];
        for j in np.arange(size[0]):#width
            it = i# - T[1]
            jt = j# - T[0]
            kt = T[2]
            #二维图像平面坐标系->摄像机坐标系
            _x = jt*iR[0, 0] +it*iR[0, 1] + iR[0, 2],
            _y = jt*iR[1, 0] +it*iR[1, 1] + iR[1, 2],
            _w = jt*iR[2, 0] +it*iR[2, 1] + iR[2, 2];

            #归一化摄像机坐标系，相当于假定在Z=1平面上
            x = _x/_w - T[0]/T[2]
            y = _y/_w - T[1]/T[2];

            #求鱼眼半球体截面半径r
            r = np.sqrt(x*x + y*y);
            #求鱼眼半球面上一点与光心的连线和光轴的夹角Theta
            theta = np.arctan(r);
            #畸变模型求出theta_d，相当于有畸变的角度值
            theta2 = theta*theta
            theta4 = theta2*theta2
            theta6 = theta4*theta2
            theta8 = theta4*theta4;
            theta_d = theta * (1 + k[0]*theta2 + k[1]*theta4 + k[2]*theta6 + k[3]*theta8);
            #利用有畸变的Theta值，将摄像机坐标系下的归一化三维坐标，重投影到二维图像平面，得到(j,i)对应畸变图像中的(u,v)
            scale = 1.0 if (r == 0) else theta_d / r
            u = f[0]*x*scale + c[0];
            v = f[1]*y*scale + c[1];

            #保存(u,v)坐标到mapx,mapy
            map1[i,j] = u
            map2[i,j] = v 

            #这三条语句是上面 ”//二维图像平面坐标系->摄像机坐标系“的一部分，是矩阵iR的第一列，这样写能够简化计算
            #_x += iR[0, 0];
            #_y += iR[1, 0];
            #_w += iR[2, 0];
        
    
    return np.array(map1,np.float32),np.array(map2,np.float32)


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

for i in np.arange(0,successImageNum):
    R = np.eye(3)
    t = image_Seq[i]
    rotation3x3,jacb = cv2.Rodrigues(-rotation_vectors[i]);

    rot4x4 = np.zeros((4,4))
    rot4x4[:3,:3] = rotation3x3
    rot4x4[3,3] = 1
    
    
    trans4x4 = np.eye(4)
    trans4x4[:3,3] = translation_vectors[i][:,0]

    
    RTMat = np.dot(rot4x4,trans4x4)
    
    #RR = np.zeros((3,3))
    #RR = cv2.estimateAffine3D(rotation_vectors[i])
    #RR = RR.rotation();

    #mapx,mapy = cv2.fisheye.initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,image_size,cv2.CV_32FC1)
    mapx,mapy = cv2.fisheye.initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,-rotation_vectors[i],intrinsic_matrix,image_size,cv2.CV_32FC1)
    #mapx1,mapy1 = pyinitUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,-rotation_vectors[i],-translation_vectors[i],intrinsic_matrix,image_size)

    
    #mapx -= translation_vectors[i][0]
    #mapy -= translation_vectors[i][1]

    
    w = t.shape[1]
    h = t.shape[0]
    f = 1

    Mat2D3D = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
    Mat3D2D = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

    crs = np.dot(Mat3D2D,Mat2D3D)
    crs = np.dot(Mat2D3D,Mat3D2D)

    M4x4 = np.dot(Mat3D2D,np.dot(trans4x4,np.dot(rot4x4,Mat2D3D)))

    #t = cv2.warpPerspective(t,M4x4,(t.shape[1],t.shape[0]))

    t = cv2.remap(t,mapx,mapy,cv2.INTER_LINEAR)



    saveFileName = '{0}{1}{2}.jpg'.format(filePath,'result_org',i+1)
    print("保存文件{0}".format(saveFileName))
    cv2.imwrite(saveFileName,t)


    imageTrans = ImageTransformation3D.ImageTransformer(saveFileName,(t.shape[0],t.shape[1],3))

    rotvec = rotation_vectors[i]
    transvec = translation_vectors[i]

    theta =  -rotvec[0][0]
    phi   =  -rotvec[1][0]
    gamma =  -rotvec[2][0]
    dx    =  transvec[0][0]  
    dy    =  transvec[1][0]  
    dz    =  transvec[2][0]  

    #img = imageTrans.rotate_along_axis(np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(gamma), dx, dy, dz)
    img = imageTrans.rotate_along_axis(0,0,0, dx, dy, dz)
    cv2.imshow('image',img)
    cv2.waitKey(0)