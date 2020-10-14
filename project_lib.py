import numpy as np
import cv2
import imutils
import math
from scipy.ndimage import label

real_colors=[ #Testes práticos, médias sucessivas
('DarkGreen', (48,87,18)),
('Green', (0,150,0)),#!
('LightGreen', (44,162,123)),
('OliveGreen', (75,99,107)),
('DarkOliveGreen', (50,110,85)), #!
('Red', (44,47,160)),
('LightViolet', (190,132,148)),
('Violet', (107,26,46)),
('DarkGray', (87,76,70)),
('LightGray', (134,126,118)),
('DarkPink', (77,30,112)),
('Pink', (131,65,174)),
('LightPink', (195,135,194)),
('Brown', (43,49,74)),
('Yellow', (48,172,216)),
('LightYellow', (120,163,178)),
('Orange', (42,100,200)),
('DarkBlue', (123,44,9)),
('LightBlue', (207,147,101)),
('Black', (50,50,50)),
('White', (210,210,210))]

#CLASSE LEGO
standard_colors={
'DarkGreen': (0,60,0),
'Green': (0,150,0),
'LightGreen': (0,255,0),
'OliveGreen': (0,130,130),
'DarkOliveGreen': (50,110,85),
'Red': (0,0,255),
'LightViolet': (150,110,220),
'Violet': (210,0,150),
'DarkGray': (100,100,100),
'LightGray': (200,200,200),
'DarkPink': (150,20,255),
'Pink': (180,110,255),
'LightPink': (190,190,255),
'Brown': (20,70,140),
'Yellow': (0,255,255),
'LightYellow': (150,255,255),
'Orange': (0,140,255),
'DarkBlue': (255,0,0),
'LightBlue': (255,210,140),
'Black': (50,50,50),
'White': (255,255,255)}

class Lego:

	def __init__(self, color, form, pontos, center):
		self.color = color
		self.form = form
		self.pontos= pontos
		self.center= center
		self.mask = 0

	def __str__(self):
		#points = [[[0 0]]] -> tmp = [(0,0)]
		if(len(self.pontos) > 4):
			tmp = []
			for p in self.pontos:
				tmp.append((p[0][0], p[0][1]))
			return self.color+str(self.form) +" "+ str( [(int(a[0]*7.56), int(a[1]*7.56)) for a in tmp] )
		else:
			return self.color+str(self.form) +" "+ str( [(int(a[0]*7.56), int(a[1]*7.56)) for a in self.pontos] )

	def set_mask(self, mask):
		self.mask = mask
	
	def get_mask(self):
		return self.mask
	
	def set_color(self, color):
		self.color = color
		
	def set_form(self, form):
		self.form = form
		
	def set_pontos(self, pontos):
		self.pontos = pontos

	def set_center(self, center):
		self.center = center

	def get_color(self):
		return self.color
	
	def get_form(self):
		return self.form
	
	def get_pontos(self):
		return self.pontos
	
	def get_center(self):
		return self.center
	
	def print_file(self):
		if(len(self.pontos) > 4):
			tmp = []
			for p in self.pontos:
				tmp.append((p[0][0], p[0][1]))
			return str(self.color + str(self.form) + ',"'+ str( [(int(a[0]*7.56), int(a[1]*7.56)) for a in tmp] ) + '"\n')
		else:
			return str(self.color + str(self.form) + ',"'+ str( [(int(a[0]*7.56), int(a[1]*7.56)) for a in self.pontos] ) + '"\n')

	def draw (self, img):
		for key in standard_colors:
			if(key==self.color):
				pts = np.array(self.pontos, np.int32)
				cv2.polylines(img, [pts], True, standard_colors[key], thickness=3)
				cv2.fillPoly(img, [pts], standard_colors[key])


#DETEÇÃO DE CENTROS
centroids = [];
def Countours(image):
	cnts= cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	return cnts

def detect_simples(c):
	global centroids
	h=cv2.moments(c)
	if h["m00"]!=0:
		cX = int(h["m10"] / h["m00"])
		cY = int(h["m01"] / h["m00"])
		centroids.append((cX,cY))

def find_centros( img ):
	global centroids
	centroids = [];
	cnts = Countours(img);
	for c in cnts:
		detect_simples(c)
	return centroids

#DETEÇÃO DE FORMAS
def dist(x,y):   
   return np.linalg.norm(x-y)
def uniques(x):
     return np.unique([(dist(x[0],x[1])*10),(dist(x[0],x[2])*10),(dist(x[0],x[3])*10),(dist(x[1],x[2])*10),(dist(x[1],x[3])*10),(dist(x[2],x[3])*10)])	
def distMin(x):
     return min(dist(x[0],x[1]),dist(x[0],x[2]),dist(x[0],x[3]),dist(x[1],x[2]),dist(x[1],x[3]),dist(x[2],x[3]))	
def distMax(x):
     return max(dist(x[0],x[1]),dist(x[0],x[2]),dist(x[0],x[3]),dist(x[1],x[2]),dist(x[1],x[3]),dist(x[2],x[3]))	
def media(x):
    return np.mean([dist(x[0],x[1]),dist(x[0],x[2]),dist(x[0],x[3]),dist(x[1],x[2]),dist(x[1],x[3]),dist(x[2],x[3])])

def detect(c):
	d=0
	d2=0	
	d_t=0
	area=0
	l=0
	# initialize the shape name and approximate the contour
	shape = "unidentified"

	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.03* peri, True)
	# if the shape is a triangle, it will have 3 vertices
	rect=cv2.minAreaRect(c) 
	box = np.int0(cv2.boxPoints(rect))
	points = box
	area=cv2.contourArea(points)
	num_points = len(points)

	if   len(approx)==5 or len(approx)==6:
		if(area>4500 and area<4520):
			shape = "Pentagono"
	
		elif(area>5450 and area<5470):
			shape = "Hexagono"
		return shape,area, c
	# if the shape has 4 vertices, it is either a square or
	# a rectangle
	elif num_points == 3:
		shape = "triangle"
	elif num_points == 4:	
		
		l=distMin(points)  #calculate side
		d=distMax(points)  #calculate diagonal
		
		d_t=l*np.sqrt(2)   #calculate theorethical diagonal corresponding to the side(l)
		
		difDist=uniques(points)
		if area > 1915 and area < 1930:
			shape="trapezio"
			return shape,area,c
		elif d-d_t >= -6 and d-d_t <=6		:
			shape = "square"
		
		else:
			shape="rectangle"
		
	
	else:# otherwise, fail
		shape = "fail"
	
	tmp = []
	for p in points:
		tmp.append((p[0], p[1]))
	return shape,area, tmp

standard_ratios = [8.2, 6.2, 4.5, 3.3, 2, 1.5, 1]
standard_areas = [11000, 7500, 5000, 1850, 1300, 1000]
#1   -> 13000_12, 5000_9, 1000_2
#1.5 -> 7500_10, 1800_3
#2   -> 2400_6, 13000_11
#3.3 -> 3800_7, 
#4.5 -> 5000_8, 1000_1
#6.2 -> x_4
#8  -> x_5
def calc_form(points):
	form = 14
	area_idx = 0
	ratio_idx = 0
	r = 0
	if(len(points) == 4):
		soma = 0
		for i in range(0, 3):
			soma += points[i][0]*points[i+1][1] - points[i][1]*points[i+1][0]
		soma += points[3][0]*points[0][1] - points[3][1]*points[0][0]
		ar = int(math.fabs( (soma)/2 ))
		side1 = dist2points(points[0], points[1])
		side2 = dist2points(points[1], points[2])
		r = max(side1/side2, side2/side1)
		closest = 8
		for i in standard_ratios: #comparar com as razãos entre lados
			d = math.fabs(r - i)
			if( d < closest ):
				closest = d
				ratio_idx = standard_ratios.index( i )
		
		closest = 13000
		if (ratio_idx == 0):
			form = 5
		if (ratio_idx == 1):
			form = 4
		if (ratio_idx == 2):
			standard_areas = [5000, 1000]
			for i in standard_areas:
				d = math.fabs(ar - i)
				if( d < closest ):
					closest = d
					area_idx = standard_areas.index( i )
			if(area_idx == 0):
				form = 8
			if(area_idx == 1):
				form = 1
		if (ratio_idx == 3):
			standard_areas = [3800, 0] #########
			for i in standard_areas:
				d = math.fabs(ar - i)
				if( d < closest ):
					closest = d
					area_idx = standard_areas.index( i )
			if(area_idx == 0):
				form = 7
			if(area_idx == 1):
				form = 0
		if (ratio_idx == 4):
			standard_areas = [2400, 13000]
			for i in standard_areas:
				d = math.fabs(ar - i)
				if( d < closest ):
					closest = d
					area_idx = standard_areas.index( i )
			if(area_idx == 0):
				form = 6
			if(area_idx == 1):
				form = 11
		if (ratio_idx == 5):
			standard_areas = [7500, 1800]
			for i in standard_areas:
				d = math.fabs(ar - i)
				if( d < closest ):
					closest = d
					area_idx = standard_areas.index( i )
			if(area_idx == 0):
				form = 10
			if(area_idx == 1):
				form = 3
		if (ratio_idx == 6):
			standard_areas = [13000, 5000, 1000]
			for i in standard_areas:
				d = math.fabs(ar - i)
				if( d < closest ):
					closest = d
					area_idx = standard_areas.index( i )
			if(area_idx == 0):
				form = 12
			if(area_idx == 1):
				form = 9
			if(area_idx == 2):
				form = 2
	return form

#REMOÇÃO DE BACKGROUND
def remove_background( img_bgr ):
	kernel = np.ones((3 ,3), np.uint8)
	#imagem para tratamento de fundo
	back_img = img_bgr.copy()
	back_img = cv2.GaussianBlur(back_img,(5,5),0)
	###########################REMOÇÃO DE FUNDO WATERSHED E FLOODFILL###########################
	'''Watershed'''
	final_mask = np.zeros((back_img.shape[0], back_img.shape[1]), dtype=np.uint8)
	for i in range(0,3):
		img_gray = back_img[:,:,i] #aplicar nos canais todos, porque parece resultar com os legos Azuis claros , Rosa e Amarelos
		_, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
		img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((3, 3), dtype=int))
		border = cv2.dilate(img_bin, None, iterations = 2)
		border = border - cv2.erode(border, None)
		dt = cv2.distanceTransform(img_bin, 2, 3)
		dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
		_, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
		lbl, ncc = label(dt)
		lbl = lbl * (255 / (ncc + 1))
		lbl[border == 255] = 255
		lbl = lbl.astype(np.int32)
		cv2.watershed(img_bgr, lbl)
		lbl[lbl == -1] = 0
		lbl = lbl.astype(np.uint8)
		areas = 255 - lbl
		water_mask = areas.copy()
		water_mask[areas == 0] = 255
		water_mask[areas != 0] = 0
		water_mask = cv2.dilate(water_mask, None)
		final_mask = final_mask | water_mask

	'''FloodFill'''
	h,w,chn = back_img.shape
	seed = (1, 1) #Ponto de referencia do FloolFill
	flood_mask = np.zeros((h+2,w+2),np.uint8)
	floodflags = 4
	floodflags |= cv2.FLOODFILL_MASK_ONLY
	floodflags |= (255 << 8)
	num,back_img,flood_mask,rect = cv2.floodFill(back_img, flood_mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)
	flood_mask = cv2.bitwise_not(flood_mask[:back_img.shape[0],:back_img.shape[1]])

	final_mask = final_mask | flood_mask
	final_mask = cv2.erode(final_mask, kernel, iterations = 1)
	return final_mask
	

#CALCULOS
def dist2points(a,b):
	return math.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )
