import cv2
import pandas as pd
import face_recognition
from os import listdir
from datetime import datetime
import streamlit as st

def detect(path, format):
    # getting the face locations in an image and their encodings
    def encode(images):
        face_location = []
        encodings = []

        if type(images) == list:
            for i in images:
                face_location.append(face_recognition.face_locations(i[1])[0])
                encodings.append(face_recognition.face_encodings(i[1])[0])
        else:
            face_location = face_recognition.face_locations(images)
            encodings = face_recognition.face_encodings(images)

        return face_location, encodings

    # loading images
    persons = []

    for file in listdir('People/'):
        name, extension = file.split(".")
        image = face_recognition.load_image_file('People/' + file)
        persons.append([name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)])

    def compare_and_distances(img, encodings):
        results = []
        faceDis = []

        img_encodings = face_recognition.face_encodings(img)

        for i in img_encodings:
            results.append(face_recognition.compare_faces(encodings, i))

            # lower the distance = better match
            faceDis.append(face_recognition.face_distance(encodings, i))

        return results, faceDis

    known_faces_encodings = encode(persons)

    if format == 'image':
        image = cv2.imread(path)
        face_loc, encodings = encode(image)

        img_det = image.copy()

        for i in face_loc:
            cv2.rectangle(img_det, (i[3], i[0]), (i[1], i[2]), (255, 0, 0), 2)

        st.write('## Original Image')
        st.image('image.jpg')

        img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)

        st.write('## Image with face detctions')
        st.image(img_det)

        #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif format == 'video' or 'webcam':
        if format == 'webcam':
            cap = cv2.VideoCapture(0) #webcam

        elif format == 'video':
            cap = cv2.VideoCapture(path)  # video
            st.write('## Playing Video')

        bool = True

        while bool:
            ret, img = cap.read()

            frame = st.empty()

            face_loc, enc = encode(img)

            if len(face_loc) != 0:
                preds, face_dist = compare_and_distances(img, known_faces_encodings[1])

                for i in range(len(preds)):
                    for j in preds[i]:
                        if j == True:
                            index = preds[i].index(j)
                            name = persons[index][0]

                            x, y, x1, y1 = face_loc[i][3], face_loc[i][0], face_loc[i][1], face_loc[i][2]
                            cv2.putText(img, name, (int(x1 + 15), int(y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (57, 255, 20), 2)
                            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)

                            df = pd.read_csv('Attendance.csv')

                            if name not in df['Name'].tolist():
                                df.loc[len(df)] = [name, datetime.now().date(), datetime.now().time()]
                                df.to_csv('Attendance.csv', index=False)

                                bool = False
                                st.success('Your Attendance has been taken')
                                break
                        else:
                            continue


            frame.image(img, channels="BGR")
            #frame.empty()
            #if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            #    break

        # kill open cv things
        #cap.release()
        #cv2.destroyAllWindows()


def main():
    st.write("# Face Recognition Attendance System")

    option = st.selectbox('Choose',('-','See Some Examples','Give Attendance Using Webcam'))

    if option == 'See Some Examples':
        format = st.radio('Select Format',('Image','Video'), index=0)

        if format == 'Image':
            detect('image.jpg', format.lower())

        elif format == 'Video':
            detect('video.mp4', format.lower())


    elif option == 'Give Attendance Using Webcam':
        start = st.button('START')

        if start:
            detect('webcam', 'webcam')



main()
