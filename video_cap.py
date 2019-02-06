"""import cv2

camera = cv2.VideoCapture(0)
i = 0
while i < 10:
    input('Press Enter to capture')
    return_value, image = camera.read()
    cv2.imwrite('opencv'+str(i)+'.png', image)
    i += 1
del(camera)"""
import base64
import cv2
from PIL import Image
import pymongo
import gridfs

# Windows dependencies
# - Python 2.7.6: http://www.python.org/download/
# - OpenCV: http://opencv.org/
# - Numpy -- get numpy from here because the official builds don't support x64:
#   http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy

# Mac Dependencies
# - brew install python
# - pip install numpy
# - brew tap homebrew/science
# - brew install opencv
connection=pymongo.MongoClient("mongodb://admin:admin123@ds161724.mlab.com:61724/webchat" )
#database=connection['webchat']
db=connection.webchat
#fs=gridfs.GridFS(db,'photo')


def change_pixel(img):
	for pixel in img.getdata():
		print(pixel)

def insert_image(img_name):
	with open(img_name,'rb') as image_file:
		encoded_string=base64.b64encode(image_file.read())
	print(encoded_string)
	if db.happy_image.find_one({'name':'happy'}):
		obj=db.happy_image.find_one({'name':'happy'})
		obj['image']=encoded_string
		db.happy_image.save(obj)
	else:
		abc=db.happy_image.insert_one({'name':'happy','image':encoded_string})
	print('operation done!')

cap = cv2.VideoCapture(0)







while(True):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    img=cv2.imshow('frame', rgb)
    cv2.imwrite('happy.jpg',frame)
    #img=Image.open('happy.jpg')
    #gfsFile=GridFSInputFile.gfsPhoto.createFile(img)
    #gfsFile.setFilename('suyash')
    #gfsFile.save()

    insert_image('happy.jpg')
    
    #stored=fs.put(img,filename='suyash')
    #outputdata=fs.get(stored).read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        #out = cv2.imwrite('capture.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()