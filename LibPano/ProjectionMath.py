import numpy as np
import cv2

class NFOV():
    def __init__(self, height=400, width=800):
        self.FOV = [0.45, 0.45]
        self.PI = np.pi
        self.PI_2 = np.pi * 0.5
        self.PI2 = np.pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) if isCenterPt else (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx, yy]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
       
        cv2.imshow('img',nfov)
        cv2.waitKey(0)
       
        return nfov

    def toNFOVbilinear(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return self._bilinear_interpolation(spericalCoord)

    def toNFOV(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)

        screen_coord = spericalCoord
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height
        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)

        #x0 = np.clip(x0,0,self.frame_width -1 )
        #y0 = np.clip(y0,0,self.frame_height -1 )

        #dst = np.zeros((self.height,self.width))
        dst = self.frame[y0,x0,:]
        
        cv2.imshow('img',dst)
        cv2.waitKey(0)

        return dst


class GnomonicProjection():
   
    def __init__(self, lamda = 0, phi = 0):
        self.lamda0 = lamda;
        self.phi0 = phi;

    def Projection(self,lamda,phi):
        cosc = np.sin(self.phi0) * np.sin(phi) + np.cos(self.phi0) * np.cos(phi) * np.cos(lamda - self.lamda0)

        x = np.cos(phi) * np.sin(lamda - self.lamda0) / cosc
        y = (np.cos(self.phi0) * np.sin(phi) - np.sin(self.phi0) * np.cos(phi) * np.cos(lamda - self.lamda0) )/cosc

        return x,y

    def InverseProjection(self,x,y):
        p = np.sqrt(x*x + y*y)
        c = np.arctan(p)

        phi = np.arcsin( np.cos(c) * np.sin(self.phi0) + y * np.sin(c) * np.cos(self.phi0)/p  )
        lamda = self.lamda0 + np.arctan2(x * np.sin(c) , (p * np.cos(self.phi0)* np.cos(c) - y * np.sin(self.phi0) * np.sin(c)))
        return lamda,phi

if __name__ == '__main__':
    
    srcimg = cv2.imread('LibPano/Images/Projection/360.jpg',cv2.IMREAD_COLOR)
    
    hedge = np.linspace(0, 19, 20)
    vedge = np.linspace(0, 9, 10)
    xx,yy = np.meshgrid(hedge,vedge)
    grid = np.array([xx.ravel(), yy.ravel()]).T

    nfov = NFOV()
    center_point = np.array([0.5, 1])  # camera center point (valid range [0,1])
    #nfov.toNFOV(srcimg, center_point)
    
    
    
    #srcimg = cv2.imread('LibPano/Images/Projection/Equirectangular.jpg',cv2.IMREAD_COLOR)

    width = 512
    height = 512

    dstimg = np.zeros((height,width))

    _GnomonicProjection = GnomonicProjection(0 ,np.pi /2)
    #_GnomonicProjection = GnomonicProjection(0 ,0)

    
    minx = 99999
    maxx = -99999
    miny = 99999
    maxy = -99999
    for lamda in np.arange(-np.pi ,np.pi,0.01):
        for phi in np.arange(-np.pi / 2 ,np.pi / 2 , 0.01):
            #x,y =  _GnomonicProjection.Projection(lamda,phi)

            #minx = np.minimum(minx,x);
            #maxx = np.maximum(maxx,x);
            #miny = np.minimum(miny,y);
            #maxy = np.maximum(maxy,y);
            pass

    
    hedge = np.linspace(-np.pi, np.pi, width) / 4
    vedge = np.linspace(-np.pi/2, np.pi/2, height) 
    xx,yy = np.meshgrid(hedge,vedge)

    lamda,phi = _GnomonicProjection.InverseProjection(xx,yy)

    #lamda = np.mod(lamda  + np.pi ,np.pi * 2) - np.pi
    #phi = np.mod(phi + np.pi / 2,np.pi ) - np.pi / 2

    lamda = np.floor((lamda / np.pi/2 + 0.5) * srcimg.shape[1]).astype(np.int32)
    phi = np.floor((phi / np.pi + 0.5) * srcimg.shape[0]).astype(np.int32)


    m = lamda > np.pi;
    lamda = lamda  +  m * -srcimg.shape[1];

    
    m = phi > np.pi/2;
    phi = phi  +  m * -srcimg.shape[0];



    dst = srcimg[phi,lamda,:]
    cv2.imshow('img',dst)
    cv2.waitKey(0)
