import cv2
import face_recognition

imgElon = face_recognition.load_image_file('ImagesBasic/e.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest1 = face_recognition.load_image_file('ImagesBasic/huang.jpg')
imgTest2 = face_recognition.load_image_file('ImagesBasic/elonTest2.jpg')
imgTest1 = cv2.cvtColor(imgTest1, cv2.COLOR_BGR2RGB)
imgTest2 = cv2.cvtColor(imgTest2, cv2.COLOR_BGR2RGB)

##Examples of comparisons
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

##Test case 1
faceLocTest1 = face_recognition.face_locations(imgTest1)[0]
encodeTest1 = face_recognition.face_encodings(imgTest1)[0]
cv2.rectangle(imgTest1, (faceLocTest1[3], faceLocTest1[0]), (faceLocTest1[1], faceLocTest1[2]), (255, 0, 255), 2)

##Test case 2
faceLocTest2 = face_recognition.face_locations(imgTest2)[0]
encodeTest2 = face_recognition.face_encodings(imgTest2)[0]
cv2.rectangle(imgTest2, (faceLocTest2[3], faceLocTest2[0]), (faceLocTest2[1], faceLocTest2[2]), (255, 0, 255), 2)

##Calculation case 1 and 2
results1 = face_recognition.compare_faces([encodeElon], encodeTest1)
results2 = face_recognition.compare_faces([encodeElon], encodeTest2)
faceDis1 = face_recognition.face_distance([encodeElon], encodeTest1)
faceDis2 = face_recognition.face_distance([encodeElon], encodeTest2)

##print out the res
print(results1, faceDis2)
cv2.putText(imgTest1, f'{results1} {round(faceDis1[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
print(results2, faceDis2)
cv2.putText(imgTest2, f'{results2} {round(faceDis2[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test1', imgTest1)
cv2.imshow('Elon Test2', imgTest2)
cv2.waitKey(0)