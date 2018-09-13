import numpy as np

import cv2

mVersion = 0;
mImageAspect = 1.0
mDistance = 0.0
mPanoHfov = 360
mCutRangeMin = -1
mCutRangeMax = -1


X_GRID_COUNT  = 180
Y_GRID_COUNT  = 90


Y_GRIDS_LENGTH = 50.0
GRID_SIDE_LENGTH  = (Y_GRIDS_LENGTH / Y_GRID_COUNT)
VERTICES_COUNT = ((X_GRID_COUNT + 1) * (Y_GRID_COUNT + 1))


ImgWidth = 512
ImgHeight = 256
channel = 3
mPanoAspect = ImgWidth/ImgHeight

mPanoHfov = 160

PI =np.pi

format = 0

srcimg = []
srcimgWidth = 0
srcimgHeight = 0


class MakeParams_t:
    scale = np.zeros((2))
    shear = np.zeros((2))
    rot = np.zeros((2))
    rad = np.zeros((6))
    mat = np.zeros((3,3))
    distance = 0
    horizontal = 0
    vertical = 0
    trans = np.zeros((5))




def move0and1(v):
    if(v < 0):
        v += 1
    if(v > 1):
        v -= 1
    return v

def erect_rect( x_dest,   y_dest,  distance):

	x_src = distance * np.arctan2(x_dest, distance);
	y_src = distance * np.arctan2(y_dest, np.sqrt(distance*distance + x_dest*x_dest));

	return x_src,y_src;



#convert erect to cartesian XYZ coordinates
def cart_erect(x_dest, y_dest, distance):
    #phi is azimuth (negative angle around y axis, starting at the z axis)
    phi = x_dest / distance;
    theta_zenith = PI / 2.0 - (y_dest / distance);
    #compute cartesian coordinates..
    #pos[2] = cos(-phi)*sin(theta_zenith);
    #pos[0] = sin(-phi)*sin(theta_zenith);
    #pos[1] = cos(theta_zenith);

    xyz = np.zeros((3))
    xyz[0] = np.sin(theta_zenith)*np.sin(phi);
    xyz[1] = np.cos(theta_zenith);
    xyz[2] = np.sin(theta_zenith)*-np.cos(phi);

    return xyz
#convert cartesian coordinates into spherical ones
def erect_cart(xyz, distance):
	x_src =np.arctan2(xyz[0], -xyz[2]) * distance;
	y_src = np.arcsin(xyz[1] / np.sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2])) * distance;

	return x_src,y_src;

#Compute intersection between line and point.
#  n : a,b,c,d coefficients of plane (a,b,c = normal vector)
#  p1: point on line
#  p2: point on line
#  See http://local.wasp.uwa.edu.au/~pbourke/geometry/planeline/

def line_plane_intersection(n, p1,p2):
    result = np.zeros((3))
    ret = 0
    i = 0;
	#direction vector of line
    d=  np.zeros((3))
    u = 0
    num = 0
    den = 0;

    for i in np.arange(0,3):
        d[i] = p2[i] - p1[i];
    num = n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2] + n[3];
    den = -n[0] * d[0] - n[1] * d[1] - n[2] * d[2];
    if (np.fabs(den) < 1e-15):
        ret = 0;
    u = num / den;

    if (u < 0):
		#This is match is in the wrong direction, ignore
        ret = 0;
	#printf("intersect, dir: %f %f %f, num: %f, denom: %f, u: %f\n", d[0], d[1], d[2], num, den, u);
    for i in np.arange(0,3):
        result[i] = p1[i] + u*d[i];
    ret = 1
    return ret,result;

# transfer a point from the master camera through a plane into camera
#  at TrX, TrY, TrZ using the plane located at Te0 (yaw), Te1 (pitch)
def plane_transfer_to_camera(x_dest, y_dest, mp):
    plane_coeff = np.zeros((4))
    p1 = np.zeros((3))
    p2 = np.zeros((3))
    intersection = np.zeros((3))

	#compute ray of sight for the current pixel in
	#the master panorama camera.
	#camera point
    p1[0] = p1[1] = p1[2] = 0;
    #point on sphere.
    p2 = cart_erect(x_dest, y_dest,  mp.distance);

	#compute plane description
    xyz = cart_erect(mp.trans[3], -mp.trans[4],	 1.0);
    plane_coeff[0:3] = xyz
	#plane_coeff[0..2] is both the normal and a point
	#on the plane.
    plane_coeff[3] = -plane_coeff[0] * plane_coeff[0] - plane_coeff[1] * plane_coeff[1] - plane_coeff[2] * plane_coeff[2];

	#printf("Plane: y:%f p:%f coefficients: %f %f %f %f, ray direction: %f %f %f\n",
	#mp->trans[3], mp->trans[4], plane_coeff[0], plane_coeff[1], plane_coeff[2], plane_coeff[3],
	#p2[0],p2[1],p2[2]);
    
    #perform intersection.
    ret, intersection = line_plane_intersection(plane_coeff, p1, p2)
    if (ret == 0):

		#printf("No intersection found, %f %f %f\n", p2[0], p2[1], p2[2]);
        return 0;

	#compute ray leading to the camera.
    intersection[0] -= mp.trans[0];
    intersection[1] -= mp.trans[1];
    intersection[2] -= mp.trans[2];

	#transform into erect
    x_src,  y_src = erect_cart(intersection, mp.distance);

	#printf("pano->plane->cam(%.1f, %.1f, %.1f, y:%1f,p:%1f): %8.5f %8.5f -> %8.5f %8.5f %8.5f -> %8.5f %8.5f\n",
    #mp->trans[0], mp->trans[1], mp->trans[2], mp->trans[3], mp->trans[4],
	#x_dest, y_dest,
	#intersection[0], intersection[1], intersection[2],
	#*x_src, *y_src);
	#*/
    return  x_src,  y_src




#Rotate equirectangular image
def rotate_erect(x_dest, y_dest, rot):
    #params: double 180degree_turn(screenpoints), double turn(screenpoints);
    x_src = x_dest + rot[1];
    while (x_src < -rot[0]):
        x_src += 2 * rot[0];
    while (x_src >  rot[0]):
        x_src -= 2 * rot[0]
    y_src = y_dest;
    return  x_src,  y_src


def sphere_tp_erect(x_dest,  y_dest,  distance):
    phi = 0
    theta = 0
    r = 0
    s = 0;
    v = np.zeros((3));
    
    phi = x_dest / distance;
    theta = -y_dest / distance + PI / 2;
    if (theta < 0):
        theta = -theta;
        phi += PI;
    if (theta > PI):
        theta = PI - (theta - PI);
        phi += PI;
    s = np.sin(theta);
    v[0] = s * np.sin(phi);	#  y' -> x
    v[1] = np.cos(theta);				#  z' -> y

    r = np.sqrt(v[1] * v[1] + v[0] * v[0]);

    theta = distance * np.arctan2(r, s * np.cos(phi));

    x_src = theta * v[0] / r;
    y_src = theta * v[1] / r;

    return  x_src,  y_src

def matrix_matrix_mult(m1, m2):

    result = np.zeros((3,3))
    for i in np.arange(3):
        for k in np.arange(3):
            result[i,k] = m1[i,0] * m2[0,k] + m1[i,1] * m2[1,k] + m1[i,2] * m2[2,k];
    return result

def matrix_inv_mult(m, vector):
    i = 0
    v0 = vector[0];
    v1 = vector[1];
    v2 = vector[2];

    for  i in np.arange(0,3):
        vector[i] = m[0,i] * v0 + m[1,i] * v1 + m[2,i] * v2;
    return vector



def persp_sphere(x_dest,  y_dest, matrix,distance):
	#params :  double Matrix[3][3], double distanceparam

    phi = 0
    theta = 0
    r = 0
    s = 0;
    v = np.zeros((3));

    r =np. sqrt(x_dest * x_dest + y_dest * y_dest);
    theta= r/ distance;
    if(r==0.0):
        s=0.0;
    else:
        s=np.sin(theta)/r;

    v[0]=s*x_dest;
    v[1]=s*y_dest;
    v[2]=np.cos(theta);

    v = matrix_inv_mult(matrix,v);
    
    r=np.sqrt(v[0]*v[0]+v[1]*v[1]);
    if(r==0.0):
        theta=0.0;
    else:
        theta= distance * np.arctan2(r,v[2])/r;
    x_src=theta*v[0];
    y_src=theta*v[1];

    
    return  x_src,  y_src

   
def resize( x_dest,  y_dest, scale):

    scale_horizontal = scale[0]
    scale_vertical = scale[1]

    x_src = x_dest * scale_horizontal
    y_src = y_dest * scale_vertical
    return  x_src,  y_src


def radial( x_dest,  y_dest, rad):

    coefficients = rad[0:4]
    scaleinput = rad[4]
    correction_radius = rad[5]

    r = 0
    scale = 1;
    
    r = (np. sqrt(x_dest*x_dest + y_dest*y_dest)) / scaleinput;
    if (r < correction_radius):
        scale = ((coefficients[3] * r + coefficients[2]) * r +	coefficients[1]) * r + coefficients[0]
    else:
        scale = 1000.0;

    x_src = x_dest * scale;
    y_src = y_dest * scale;
    return  x_src,  y_src


def vert( x_dest,  y_dest, shift):
    x_src = x_dest;
    y_src = y_dest + shift;
    return  x_src,  y_src


def horiz( x_dest,  y_dest, shift):
    x_src = x_dest + shift;
    y_src = y_dest;
    return  x_src,  y_src



def SetMatrix(a, b, c, cl):
    mx = np.zeros((3,3))
    my = np.zeros((3,3))
    mz = np.zeros((3,3))
    dummy = np.zeros((3,3))

	#Calculate Matrices;
    mx[0,0] = 1.0; 
    mx[0,1] = 0.0;
    mx[0,2] = 0.0;
    
    mx[1,0] = 0.0;
    mx[1,1] = np.cos(a);
    mx[1,2] =  np.sin(a);
    
    mx[2,0] = 0.0;
    mx[2,1] = -mx[1,2];
    mx[2,2] = mx[1,1];
    
    my[0,0] =  np.cos(b);    
    my[0,1] = 0.0;
    my[0,2] = - np.sin(b);
    
    my[1,0] = 0.0;     
    my[1,1] = 1.0;      
    my[1,2] = 0.0;

    my[2,0] = -my[0,2];	
    my[2,1] = 0.0; 
    my[2,2] = my[0,0];
    
    mz[0,0] =  np.cos(c);
    mz[0,1] =  np.sin(c);
    mz[0,2] = 0.0;

    mz[1,0] = -mz[0,1];
    mz[1,1] = mz[0,0];
    mz[1,2] = 0.0;
   
    mz[2,0] = 0.0; 
    mz[2,1] = 0.0;
    mz[2,2] = 1.0;
    if (cl > 0):
        dummy = matrix_matrix_mult(mz, mx);
    else:
        dummy = matrix_matrix_mult(mx, mz);
    m = matrix_matrix_mult(dummy, my);
    return m;





def smallestRoot(p):
    root = np.zeros((3))
    sroot = 1000.0;
    
    n, root = cubeZero(p);
    
    for i in np.arange(0,n):
		#PrintError("Root %d = %lg", i,root[i]);
        if (root[i] > 0.0 and root[i] < sroot):
            sroot = root[i];
    #PrintError("Smallest Root  = %lg", sroot);
    return sroot




def cubeZero(a):

    root = np.zeros((3))
    if (a[3] == 0.0):
	    #second order polynomial
        n, root  = squareZero(a);
    else:
        p = ((-1.0 / 3.0) * (a[2] / a[3]) * (a[2] / a[3]) + a[1] / a[3]) / 3.0;
        q = ((2.0 / 27.0) * (a[2] / a[3]) * (a[2] / a[3]) * (a[2] / a[3]) - (1.0 / 3.0) * (a[2] / a[3]) * (a[1] / a[3]) + a[0] / a[3]) / 2.0;
        if (q*q + p*p*p >= 0.0):
            n = 1;
            root[0] = cubeRoot(-q + np.sqrt(q*q + p*p*p)) + cubeRoot(-q - np.sqrt(q*q + p*p*p)) - a[2] / (3.0 * a[3]);
        else:
            phi = np.arcacos(-q / np.sqrt(-p*p*p));
            n = 3;
            root[0] = 2.0 *  np.sqrt(-p) *  np.cos(phi / 3.0) - a[2] / (3.0 * a[3]);
            root[1] = -2.0 *  np.sqrt(-p) *  np.cos(phi / 3.0 + PI / 3.0) - a[2] / (3.0 * a[3]);
            root[2] = -2.0 *  np.sqrt(-p) *  np.cos(phi / 3.0 - PI / 3.0) - a[2] / (3.0 * a[3]);
	#PrintError("%lg, %lg, %lg, %lg root = %lg", a[3], a[2], a[1], a[0], root[0]);
    return n,root

def squareZero(a):
    if (a[2] == 0.0):
	    #linear equation
        if (a[1] == 0.0):
		    #constant
            if (a[0] == 0.0):
                n = 1; root[0] = 0.0
            else:		
                n = 0;
        else:
            n = 1; root[0] = -a[0] / a[1];
    else:
        if (4.0 * a[2] * a[0] > a[1] * a[1]):
            n = 0;
        else:
            n = 2;
            root[0] = (-a[1] + sqrt(a[1] * a[1] - 4.0 * a[2] * a[0])) / (2.0 * a[2]);
            root[1] = (-a[1] - sqrt(a[1] * a[1] - 4.0 * a[2] * a[0])) / (2.0 * a[2]);
    return n, root


def cubeRoot(x):

	if (x == 0.0):
		return 0.0;
	elif (x > 0.0):
		return np.power(x, 1.0 / 3.0);
	else:
		return -np.power(-x, 1.0 / 3.0);




def SetCorrectionRadius(rad, num):
    
    a = np.zeros((4))
    for  k in np.arange(4):
        a[k] = 0.0; #1.0e-10;
        if (rad[k] != 0.0):
            a[k] = (k + 1) * rad[k];
    result = smallestRoot(a);
    return result;




def setCalicationParm(panoAspect,  panoHfov,  width,  height,  v,  a,  b,  c,	 d,  e,  y,  p,  r,  x,  u,  z,  m,  n):
    if (format == 0 and panoHfov > 179):
        panoHfov = 179

    mp = MakeParams_t()
    mp.horizontal = d;
    mp.vertical = e;

    radV = np.deg2rad(v);
    pb = np.deg2rad(panoHfov);

    mp.mt = SetMatrix(-np.deg2rad(p),0.0,-np.deg2rad(r),0);

    #distance
	#scale
    if (format == 0):
        mp.distance = panoAspect / (2.0 * np.tan(pb / 2.0));
    else:
        mp.distance = panoAspect / pb;
    mp.scale[0] = mp.scale[1] = width / radV / mp.distance;

	#shear
    mp.shear[0] = m;# im->cP.shear_x / image_selection_height;
    mp.shear[1] = n;# im->cP.shear_y / image_selection_width;
	#rot
    mp.rot[0] = mp.distance * PI;
    mp.rot[1] = -y *  mp.distance * PI / 180.0;            # rotation angle in screenpoints

    mp.rad[1] = c;
    mp.rad[2] = b;
    mp.rad[3] = a;
    mp.rad[0] = 1 - (a + b + c);
    mp.rad[5] = SetCorrectionRadius(mp.rad, 4);

    if (width < height):
        mp.rad[4] = width / 2.0
    else:
        mp.rad[4] = height / 2.0

    mp.trans[0] = x;
    mp.trans[1] = u;
    mp.trans[2] = z;
    mp.trans[3] = 0;
    mp.trans[4] = 0;

    return mp


def process(mp):
    img = np.zeros((ImgHeight,ImgWidth,channel),np.uint8)

    for r in np.arange(0,ImgHeight):
        for c in np.arange(0,ImgWidth):
            
            x_dest = c
            y_dest = r

            #x_dest = 0.555555582
            #y_dest = 0


            y_dest = (1 -y_dest / ImgHeight) - 0.5
            x_dest =  x_dest / ImgHeight - (mPanoAspect * 0.5)


            x_src = x_dest
            y_src = y_dest


            if (format == 0):
                x_src,y_src = erect_rect(x_src ,y_src,mp.distance)
            
            if (np.fabs(mp.trans[0]) > 1e-6 or np.fabs(mp.trans[1]) > 1e-6 or np.fabs(mp.trans[2]) > 1e-6):
                x_src,y_src = plane_transfer_to_camera(x_src ,y_src,mp)
            x_src,y_src = rotate_erect(x_src ,y_src,mp.rot)
            x_src,y_src = sphere_tp_erect(x_src ,y_src,mp.distance)
            x_src,y_src = persp_sphere(x_src ,y_src,mp.mt,mp.distance)
            x_src,y_src = resize(x_src ,y_src,mp.scale)
            x_src,y_src = radial(x_src ,y_src,mp.rad)

            
            x_src,y_src = vert(x_src ,y_src,mp.vertical)
            x_src,y_src = horiz(x_src ,y_src,mp.horizontal)
            if (np.fabs(mp.shear[0]) > 1e-6 or np.fabs(mp.shear[1]) > 1e-6):
                x_src,y_src = shear(x_src ,y_src,mp.shear)
            
                
            x_src += (mImageAspect * 0.5);
            y_src += 0.5;

            x_src = x_src / mImageAspect;
            #y_src = 1 - y_src

            #print(x_src,y_src)

            x_src = move0and1(x_src)
            y_src = move0and1(y_src)
            

            srcy = np.int(y_src * srcimgHeight)
            srcx = np.int(x_src * srcimgWidth)

            srcy = np.clip(srcy,0,srcimgHeight-1)
            srcx = np.clip(srcx,0,srcimgWidth-1)
            
            color =  srcimg[srcy,srcx,:];
            img[r,c,:] = color

            
        print('row:' + str(r))




    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def processtest():

    srcimg = cv2.imread('LibPano/Images/PanoCar/test.jpg',cv2.IMREAD_COLOR)
    srcimgWidth = srcimg.shape[1]
    srcimgHeight = srcimg.shape[0]
    img = np.zeros((ImgHeight,ImgWidth,channel),np.uint8)

    for r in np.arange(0,ImgHeight):
        for c in np.arange(0,ImgWidth):
            
            x_dest = c
            y_dest = r

            #x_dest = 0.555555582
            #y_dest = 0


            y_dest = (1 -y_dest / ImgHeight) - 0.5
            x_dest =  x_dest / ImgHeight - (mPanoAspect * 0.5)


            x_src = x_dest
            y_src = y_dest


            #if (format == 0):
            #    x_src,y_src = erect_rect(x_src ,y_src,mp.distance)
            
            #if (np.fabs(mp.trans[0]) > 1e-6 or np.fabs(mp.trans[1]) > 1e-6 or np.fabs(mp.trans[2]) > 1e-6):
            #    x_src,y_src = plane_transfer_to_camera(x_src ,y_src,mp)
            #x_src,y_src = rotate_erect(x_src ,y_src,mp.rot)
            #x_src,y_src = sphere_tp_erect(x_src ,y_src,mp.distance)
            #x_src,y_src = persp_sphere(x_src ,y_src,mp.mt,mp.distance)
            #x_src,y_src = resize(x_src ,y_src,mp.scale)
            #x_src,y_src = radial(x_src ,y_src,mp.rad)

            
            #x_src,y_src = vert(x_src ,y_src,.5)
            #x_src,y_src = horiz(x_src ,y_src,.5)
            #if (np.fabs(mp.shear[0]) > 1e-6 or np.fabs(mp.shear[1]) > 1e-6):
            #    x_src,y_src = shear(x_src ,y_src,mp.shear)
            
                
            x_src += (mImageAspect * 0.5);
            y_src += 0.5;

            x_src = x_src / mImageAspect;
            y_src = 1 - y_src

            #print(x_src,y_src)

            x_src = move0and1(x_src)
            y_src = move0and1(y_src)
            

            srcy = np.int(y_src * srcimgHeight)
            srcx = np.int(x_src * srcimgWidth)

            srcy = np.clip(srcy,0,srcimgHeight-1)
            srcx = np.clip(srcx,0,srcimgWidth-1)
            
            color =  srcimg[srcy,srcx,:];
            img[r,c,:] = color

            
        print('row:' + str(r))




    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def TransferParam( v,  a,  b,  c,  d,  e, y,  p,  r,  s,  t, f, g, h,  i,  j,  k,  x,  u,  z,  m, n): 
    
    mp = setCalicationParm(mPanoAspect,mPanoHfov,mImageAspect,1,v, a, b, c, d / 1024.0, e / 1024.0, y, p, r, x, u, z, m / 1024.0, n / 1024.0)
    return mp
if __name__ == '__main__':
    format = 0
    srcimg = cv2.imread('LibPano/Images/PanoCar/3.jpg',cv2.IMREAD_COLOR)
    srcimgWidth = srcimg.shape[1]
    srcimgHeight = srcimg.shape[0]
    
    mImageAspect = srcimgWidth / srcimgHeight

    mp = TransferParam(194.6728, -0.0702, 0.1738, 0, 10.0746, -62.5627, 0.6851, 45.5188, 356.3208, 55, 0, 0.2352, 0.214, 0, 0, 0, 0, 0.1528, -0.9583, 0, 0, 0)
    #process(mp)

    processtest()