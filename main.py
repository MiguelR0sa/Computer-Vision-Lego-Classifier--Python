import numpy as np
import cv2
import imutils
from scipy.spatial import distance
import sys
import math
import project_lib as pl

#INPUT DA IMAGEM
if( len(sys.argv) != 4 ):
	print("Error, expected 3 input arguments")
	print("1- input image, 2- ground truth name, 3- text file name")
	exit(1)
try:
	img_original = cv2.imread(sys.argv[1])
	ground_truth_name = sys.argv[2]
	text_file_name = sys.argv[3]
	tamanho_original = img_original.shape
except:
	print("Something isn't right!")
	print("Please, check your arguments")
	exit(2)

#Preparação da imagem
r = int(tamanho_original[0]/tamanho_original[1])
ratio_ori = tamanho_original[1]/400
r_w = 400
r_h = int(r_w*r)

img_bgr = cv2.resize(img_original, (r_w, r_h) )

#Separação em varios tipos: HSV, YUV,...
to_draw = img_bgr.copy()
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)#GRAY
img_luv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Luv) #LUV
img_yuv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV) #YUV
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) #HSV

#Remoção do background
mask_noBackground = pl.remove_background(img_bgr)

#Equilibrar a luz
mask_Background = cv2.bitwise_not(mask_noBackground)
m1, m2, m3, _ = cv2.mean(img_yuv, mask=mask_Background) #media do Backgound do YUV
diff = m1-190
img_yuv = img_yuv.astype(np.uint16)
img_yuv[:,:,0] = img_yuv[:,:,0]-diff
img_yuv[img_yuv > 255 ] = 255
img_yuv[:,:,0] = cv2.GaussianBlur(img_yuv[:,:,0],(9,9), 0)
img_yuv = img_yuv.astype(np.uint8)
img_bgr  = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

#Separar alguns legos que possam estar juntos
kernel = np.ones((5,5), np.uint8)

separarLegos_bgr = cv2.GaussianBlur(img_bgr,(7,7), 0)#BGR
img_bgr_edges = cv2.Canny(separarLegos_bgr,50,200,apertureSize = 3)
img_bgr_edges = cv2.dilate(img_bgr_edges, kernel, iterations=1)

mask_legosSeparados = cv2.bitwise_and(mask_noBackground, mask_noBackground, mask =(255-img_bgr_edges))
mask_legosSeparados = cv2.erode(mask_legosSeparados, kernel, iterations=1)
mask_legosSeparados = cv2.dilate(mask_legosSeparados, kernel, iterations=1)

centros = pl.find_centros( mask_legosSeparados )

#Recolher os legos individualmente
maskArray_legosIndividuais = []
h,w = mask_legosSeparados.shape
for c in centros:
	seed = c #Ponto de referencia do FloolFill
	flood_mask = np.zeros((h+2,w+2),np.uint8)
	floodflags = 4
	floodflags |= cv2.FLOODFILL_MASK_ONLY
	floodflags |= (255 << 8)
	_,_,flood_mask,_ = cv2.floodFill(mask_legosSeparados, flood_mask, seed, 255, 0, 0, floodflags)
	flood_mask = cv2.erode(flood_mask, None)
	flood_mask = flood_mask[1:h+1,1:w+1]
	maskArray_legosIndividuais.append(flood_mask)

#Identificação de cantos e centros / Inicializar os Legos
legos = []

i=0
for lego_mask in maskArray_legosIndividuais:
	i += 1
	cnts=pl.Countours( lego_mask );
	for c in cnts:
		shape, area, points = pl.detect(c)
		centro = pl.find_centros( lego_mask )[0] #so 1
		if(area < 100000):
			legos.append( pl.Lego("", 0, points, centro) )

forMask_YUV = cv2.GaussianBlur(cv2.bitwise_and(img_yuv,img_yuv, mask=mask_noBackground),(5,5), 0)
h,w, ch = img_bgr.shape
idx = 0
to_delete = []
for lego in legos:
	seed = lego.get_center() #Ponto de referencia do FloolFill
	flood_mask = np.zeros((h+2,w+2),np.uint8)
	floodflags = 4
	floodflags |= cv2.FLOODFILL_MASK_ONLY
	floodflags |= (255 << 8)
	num,forMask_YUV,flood_mask,rect = cv2.floodFill(forMask_YUV, flood_mask, seed, (0,0,255), (5,2,2), (5,2,2), floodflags)
	flood_mask = cv2.erode(flood_mask, None)
	flood_mask = flood_mask[:h,:w] #adiciona 2 linhas e 2 colunas então temos de retira-las
	#cv2.imshow("lego"+str(lego), flood_mask)
	
	lego.set_mask(flood_mask)
	centros = pl.find_centros( flood_mask )
	new_centro = centros[0] #so 1
	if(pl.dist2points( new_centro, lego.get_center() ) > 10):
		lego.set_center(new_centro)
	
	'''DELETE FAKE DETECTIONS'''
	for i in range(0, len(legos)):
		if( idx != i):
			if( pl.dist2points( lego.get_center(), legos[i].get_center() ) < 10):
				to_delete.append(i)
	idx += 1

legos = np.delete(legos, to_delete)

print("Legos encontrados: " + str(len(legos)))

'''ANALISE DE COR'''
forColor_BGR =cv2.GaussianBlur(cv2.bitwise_and(img_bgr,img_bgr, mask=mask_noBackground),(5,5), 0)
for lego in legos:
	m1, m2, m3, _ = cv2.mean(forColor_BGR, mask=lego.get_mask())
	minim = 255
	min_idx = 0
	for color in pl.real_colors:
		dist = math.sqrt( (m1-color[1][0])**2 + (m2-color[1][1])**2 + (m3-color[1][2])**2)
		if(dist < minim):
			minim = dist
			min_idx = pl.real_colors.index( color )
	lego.set_color( pl.real_colors[min_idx][0] )
	cv2.putText(to_draw, pl.real_colors[min_idx][0], lego.get_center(), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1, cv2.LINE_AA, False)

'''DESENHAR / MOSTRAR / ESCREVER'''
GROUND_TRUTH = np.zeros(img_bgr.shape,dtype='uint8')


content = []
try:
	f = open(text_file_name, 'r')
	content = f.readlines()
	f.close()
except:
	print("O ficheiro de texto não existe\nVamos criar um...")


content.append(ground_truth_name[0:len(ground_truth_name)-4]+":\n")
for lego in legos:
	lego.set_form( pl.calc_form(lego.get_pontos()) )
	lego.draw(GROUND_TRUTH)
	content.append(lego.print_file())
	print(lego)

content.append("\n")

f = open(text_file_name, 'w')
f.writelines(content)
f.close()

#cv2.imshow("output", GROUND_TRUTH)
#cv2.imshow("deteções", to_draw)

GROUND_TRUTH=cv2.resize(GROUND_TRUTH, (tamanho_original[1], tamanho_original[0]))
cv2.imwrite(ground_truth_name, GROUND_TRUTH)

cv2.waitKey(0)
