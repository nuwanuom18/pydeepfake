# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 20:36:29 2021

@author: Nuwan Abeynayaeke
"""


class DeepFake:
    def __init__(self ,camera_source=0):
        from importlib import resources
        import io
        self.camera_source = camera_source
        self.text = ""
        name1= "1.jpg"
        with.resources.open_binary('pydeepfake', name1) as src:
            src_image = src.read()
        src_image = io.BytesIO(src_image)
        self.source_image = src_image
        with.resources.open_binary('pydeepfake', '2.jpg') as des_i:
            des_im = des_i.read()
        des_im = io.BytesIO(des_im)
        self.destination_image = des_im
        
    
    def source_image(self , source_image):
        self.source_image = source_image
        
    def destination_image(self , destination_image):
        self.destination_image = destination_image
        
    def destination_video(self , destination_video):
        self.destination_video = destination_video

    def testfunction(self):
        import numpy as np
        print("Hello package importing is ok")  
    
    def swap_face_realtime(self):
        import cv2
        import numpy as np
        import dlib
        with.resources.open_binary('pydeepfake', 'shape_predictor_68_face_landmarks.dat') as dataset:
            ds = dataset.read()
        ds = io.BytesIO(ds)
        face_detector = dlib.get_frontal_face_detector()
        face_predictor = dlib.shape_predictor(ds)
        
        try:
            
            source_image = cv2.imread(self.source_image)
            gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
            source_canvas = np.zeros_like(gray_source)
            source_image_copy = source_image
            camera_output = cv2.VideoCapture(self.camera_source)
            
            while True:
                
                _ , frame = camera_output.read()
                
                des_image = frame
                
                gray_des = cv2.cvtColor(des_image, cv2.COLOR_BGR2GRAY)
                
                
                des_canvas = np.zeros_like(gray_des)
                
                des_canvas = np.zeros(des_image.shape, np.uint8)
                
                des_faces = face_detector(gray_des) 
                
                if len(des_faces) >= 1 :
                
                    for face in des_faces:
                        
                        des_landmark = face_predictor(gray_des , face)
                        
                        des_landmark_points = []
                        
                        for n in range(68):
                            des_landmark_points.append((des_landmark.part(n).x,des_landmark.part(n).y))
                        
                        des_landmark_points_arr  = np.array(des_landmark_points)
                        
                        #convex of des image
                        des_convex = cv2.convexHull(des_landmark_points_arr)    
                        
                    source_faces = face_detector(gray_source)
                    
                    for face in source_faces:
                        
                        source_landmark = face_predictor(gray_source , face)
                        
                        source_landmark_points = []
                        
                        for n in range(68):
                            source_landmark_points.append([source_landmark.part(n).x,source_landmark.part(n).y])
                        
                        source_landmark_points_arr  = np.array(source_landmark_points)
                        
                        #convex of source image
                        source_convex = cv2.convexHull(source_landmark_points_arr)      
                        
                        cv2.fillConvexPoly(source_canvas, source_convex, 255)
                        
                        source_face = cv2.bitwise_and(source_image, source_image, mask = source_canvas)
                        
                        rect = cv2.boundingRect(source_convex)
                        
                        subdiv = cv2.Subdiv2D(rect)
                        
                        subdiv.insert(source_landmark_points)
                        
                        triangles = np.array(subdiv.getTriangleList())
                        
                        triangle_indexes = []
                        
                        for triangle_points in triangles:
                            p1 = (triangle_points[0] , triangle_points[1])
                            p2 = (triangle_points[2] , triangle_points[3])
                            p3 = (triangle_points[4] , triangle_points[5])
                            
                            triangle = [np.where((source_landmark_points_arr == p1).all(axis =1))[0][0] , np.where((source_landmark_points_arr == p2).all(axis =1))[0][0], np.where((source_landmark_points_arr == p3).all(axis =1))[0][0]]
                            
                            triangle_indexes.append(triangle)
                        
                    for triangle_index in triangle_indexes:
                        
                        source_triangle = np.array([source_landmark_points[triangle_index[0]], source_landmark_points[triangle_index[1]], source_landmark_points[triangle_index[2]]])
                        
                        source_rect = cv2.boundingRect(source_triangle)
                        
                        x, y , w , h = source_rect
                        
                        source_cropped_rect = source_image[y: y+h , x: x+w]
                        
                        source_triangle_points = np.array([[source_landmark_points[triangle_index[0]][0]-x , source_landmark_points[triangle_index[0]][1]-y] , [source_landmark_points[triangle_index[1]][0]-x , source_landmark_points[triangle_index[1]][1]-y], [source_landmark_points[triangle_index[2]][0]-x , source_landmark_points[triangle_index[2]][1]-y]])
                        
                        des_triangle = np.array([des_landmark_points[triangle_index[0]], des_landmark_points[triangle_index[1]], des_landmark_points[triangle_index[2]]])
                        
                        source_triangle_points = np.float32(source_triangle_points)
                        
                        des_rect = cv2.boundingRect(des_triangle)
                        
                        x, y, w, h = des_rect
                        
                        des_cropped_rect = source_image[y:y+h , x:x+w]
                        
                        des_cropped_rect_mask = np.zeros((h,w), np.uint8)
                        
                        des_triangle_points = np.array([[des_landmark_points[triangle_index[0]][0]-x , des_landmark_points[triangle_index[0]][1]-y] , [des_landmark_points[triangle_index[1]][0]-x , des_landmark_points[triangle_index[1]][1]-y], [des_landmark_points[triangle_index[2]][0]-x , des_landmark_points[triangle_index[2]][1]-y]])
                       
                        
                        cv2.fillConvexPoly(des_cropped_rect_mask, des_triangle_points, 255)
                        
                        des_triangle_points = np.float32(des_triangle_points)
                        
                        mat = cv2.getAffineTransform(source_triangle_points, des_triangle_points)
                        
                        warped_affine_triangle = cv2.warpAffine(source_cropped_rect, mat, (w,h))
                    
                        wraped_affine_triangle = cv2.bitwise_and(warped_affine_triangle, warped_affine_triangle, mask = des_cropped_rect_mask)
                        
                        new_canvas_piece = des_canvas[y:y+h , x:x+w]
                        
                        des_canvas_area = cv2.cvtColor(des_canvas[y:y+h , x:x+w], cv2.COLOR_BGR2GRAY)
                        
                        _ , mask_created_triangle = cv2.threshold(des_canvas_area , 1, 255, cv2.THRESH_BINARY_INV)
                        
                        warped_affine_triangle = cv2.bitwise_and(wraped_affine_triangle, wraped_affine_triangle, mask = mask_created_triangle)
                           
                        new_canvas_piece = cv2.add(new_canvas_piece, warped_affine_triangle)
                        
                        des_canvas[y:y+h, x:x+w] = new_canvas_piece
                    
                    new_des_canvas = np.zeros_like(gray_des)
                    face_mask = cv2.fillConvexPoly(new_des_canvas, des_convex, 255)
                    new_des_canvas = cv2.bitwise_not(face_mask)
                    
                    filled_cover = cv2.bitwise_and(des_image, des_image , mask = new_des_canvas)
                    filled_cover_and_face = cv2.add(filled_cover, des_canvas)
                        
                    x,y,w,h = cv2.boundingRect(des_convex)
                    face_center = (int((x+x+w)/2), int((y+y+h)/2))
                    
                    
                    cloned_face = cv2.seamlessClone(filled_cover_and_face, des_image, face_mask, face_center, cv2.NORMAL_CLONE)
                    
                    cv2.namedWindow("DEEPFAKEOUTPUT", cv2.WINDOW_NORMAL) 
                    
                    resized_output = cv2.resize(cloned_face, (900, 500))  
                    
                    cv2.imshow("DEEPFAKEOUTPUT", resized_output)
                    
                else:
                    cv2.namedWindow("DEEPFAKEOUTPUT", cv2.WINDOW_NORMAL) 
                    
                    resized_output = cv2.resize(des_image, (900, 500))  
                    
                    #cv2.putText(cloned_face,'Hack Projects',(10,500), fontcv2.FONT_HERSHEY_SIMPLEX , 1,(255,255,255),2)
                    
                    cv2.imshow("DEEPFAKEOUTPUT", resized_output)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break   
            
                
            camera_output.release() 
            cv2.destroyAllWindows()  
        except:
            print('error: Failed to access camera')
            
    def put_bottom_text(self, text):
        self.text = text
        
    def swap_image_faces(self):
        import cv2
        import numpy as np
        import dlib
        with.resources.open_binary('pydeepfake', 'shape_predictor_68_face_landmarks.dat') as dataset:
            ds = dataset.read()
        ds = io.BytesIO(ds)
        
        face_detector = dlib.get_frontal_face_detector()
        face_predictor = dlib.shape_predictor(ds)
        
        source_image = cv2.imread(self.source_image)
        des_image = cv2.imread(self.destination_image)
        
        source_image_copy = source_image
        des_image_copy = des_image
        
        gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        gray_des = cv2.cvtColor(des_image, cv2.COLOR_BGR2GRAY)
        
        source_canvas = np.zeros_like(gray_source)
        des_canvas = np.zeros_like(gray_des)
        
        des_canvas = np.zeros(des_image.shape, np.uint8)
        
        des_faces = face_detector(gray_des) 
        
        for face in des_faces:
            
            des_landmark = face_predictor(gray_des , face)
            
            des_landmark_points = []
            
            for n in range(68):
                des_landmark_points.append((des_landmark.part(n).x,des_landmark.part(n).y))
            
            des_landmark_points_arr  = np.array(des_landmark_points)
            
            #convex of des image
            des_convex = cv2.convexHull(des_landmark_points_arr)    
            
        source_faces = face_detector(gray_source)
        
        for face in source_faces:
            
            source_landmark = face_predictor(gray_source , face)
            
            source_landmark_points = []
            
            for n in range(68):
                source_landmark_points.append([source_landmark.part(n).x,source_landmark.part(n).y])
            
            source_landmark_points_arr  = np.array(source_landmark_points)
            
            #convex of source image
            source_convex = cv2.convexHull(source_landmark_points_arr)      
            
            cv2.fillConvexPoly(source_canvas, source_convex, 255)
            
            source_face = cv2.bitwise_and(source_image, source_image, mask = source_canvas)
            
            rect = cv2.boundingRect(source_convex)
            
            subdiv = cv2.Subdiv2D(rect)
            
            subdiv.insert(source_landmark_points)
            
            triangles = np.array(subdiv.getTriangleList())
            
            triangle_indexes = []
            
            for triangle_points in triangles:
                p1 = (triangle_points[0] , triangle_points[1])
                p2 = (triangle_points[2] , triangle_points[3])
                p3 = (triangle_points[4] , triangle_points[5])
                
                triangle = [np.where((source_landmark_points_arr == p1).all(axis =1))[0][0] , np.where((source_landmark_points_arr == p2).all(axis =1))[0][0], np.where((source_landmark_points_arr == p3).all(axis =1))[0][0]]
                
                triangle_indexes.append(triangle)
            
        for triangle_index in triangle_indexes:
            
            source_triangle = np.array([source_landmark_points[triangle_index[0]], source_landmark_points[triangle_index[1]], source_landmark_points[triangle_index[2]]])
            
            source_rect = cv2.boundingRect(source_triangle)
            
            x, y , w , h = source_rect
            
            source_cropped_rect = source_image[y: y+h , x: x+w]
            
            source_triangle_points = np.array([[source_landmark_points[triangle_index[0]][0]-x , source_landmark_points[triangle_index[0]][1]-y] , [source_landmark_points[triangle_index[1]][0]-x , source_landmark_points[triangle_index[1]][1]-y], [source_landmark_points[triangle_index[2]][0]-x , source_landmark_points[triangle_index[2]][1]-y]])
            
            des_triangle = np.array([des_landmark_points[triangle_index[0]], des_landmark_points[triangle_index[1]], des_landmark_points[triangle_index[2]]])
            
            source_triangle_points = np.float32(source_triangle_points)
            
            des_rect = cv2.boundingRect(des_triangle)
            
            x, y, w, h = des_rect
            
            des_cropped_rect = source_image[y:y+h , x:x+w]
            
            des_cropped_rect_mask = np.zeros((h,w), np.uint8)
            
            des_triangle_points = np.array([[des_landmark_points[triangle_index[0]][0]-x , des_landmark_points[triangle_index[0]][1]-y] , [des_landmark_points[triangle_index[1]][0]-x , des_landmark_points[triangle_index[1]][1]-y], [des_landmark_points[triangle_index[2]][0]-x , des_landmark_points[triangle_index[2]][1]-y]])
           
            
            cv2.fillConvexPoly(des_cropped_rect_mask, des_triangle_points, 255)
            
            des_triangle_points = np.float32(des_triangle_points)
            
            mat = cv2.getAffineTransform(source_triangle_points, des_triangle_points)
            
            warped_affine_triangle = cv2.warpAffine(source_cropped_rect, mat, (w,h))
        
            wraped_affine_triangle = cv2.bitwise_and(warped_affine_triangle, warped_affine_triangle, mask = des_cropped_rect_mask)
            
            new_canvas_piece = des_canvas[y:y+h , x:x+w]
            
            des_canvas_area = cv2.cvtColor(des_canvas[y:y+h , x:x+w], cv2.COLOR_BGR2GRAY)
            
            _ , mask_created_triangle = cv2.threshold(des_canvas_area , 1, 255, cv2.THRESH_BINARY_INV)
            
            warped_affine_triangle = cv2.bitwise_and(wraped_affine_triangle, wraped_affine_triangle, mask = mask_created_triangle)
               
            new_canvas_piece = cv2.add(new_canvas_piece, warped_affine_triangle)
            
            des_canvas[y:y+h, x:x+w] = new_canvas_piece
        
        new_des_canvas = np.zeros_like(gray_des)
        face_mask = cv2.fillConvexPoly(new_des_canvas, des_convex, 255)
        new_des_canvas = cv2.bitwise_not(face_mask)
        
        filled_cover = cv2.bitwise_and(des_image, des_image , mask = new_des_canvas)
        filled_cover_and_face = cv2.add(filled_cover, des_canvas)
            
        x,y,w,h = cv2.boundingRect(des_convex)
        face_center = (int((x+x+w)/2), int((y+y+h)/2))
        
        cloned_face = cv2.seamlessClone(filled_cover_and_face, des_image, face_mask, face_center, cv2.NORMAL_CLONE)
        
        cv2.imshow("DEEPFAKE OUTPUT", cloned_face)
        
        cv2.waitKey(0)   
        cv2.destroyAllWindows()    

    def swap_video_face(self):
    
        import cv2
        import numpy as np
        import dlib
        
        
        face_detector = dlib.get_frontal_face_detector()
        face_predictor = dlib.shape_predictor('datasets/shape_predictor_68_face_landmarks.dat')
        
        
        
        try:
            
            source_image = cv2.imread(self.source_image)
            gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
            source_canvas = np.zeros_like(gray_source)
            source_image_copy = source_image
            video_output = cv2.VideoCapture(self.destination_video)
            
            while True:
                
                _ , frame = video_output.read()
                
                des_image = frame
                
                gray_des = cv2.cvtColor(des_image, cv2.COLOR_BGR2GRAY)
                
                
                des_canvas = np.zeros_like(gray_des)
                
                des_canvas = np.zeros(des_image.shape, np.uint8)
                
                des_faces = face_detector(gray_des) 
                
                if len(des_faces) >= 1 :
                
                    for face in des_faces:
                        
                        des_landmark = face_predictor(gray_des , face)
                        
                        des_landmark_points = []
                        
                        for n in range(68):
                            des_landmark_points.append((des_landmark.part(n).x,des_landmark.part(n).y))
                        
                        des_landmark_points_arr  = np.array(des_landmark_points)
                        
                        #convex of des image
                        des_convex = cv2.convexHull(des_landmark_points_arr)    
                        
                    source_faces = face_detector(gray_source)
                    
                    for face in source_faces:
                        
                        source_landmark = face_predictor(gray_source , face)
                        
                        source_landmark_points = []
                        
                        for n in range(68):
                            source_landmark_points.append([source_landmark.part(n).x,source_landmark.part(n).y])
                        
                        source_landmark_points_arr  = np.array(source_landmark_points)
                        
                        #convex of source image
                        source_convex = cv2.convexHull(source_landmark_points_arr)      
                        
                        cv2.fillConvexPoly(source_canvas, source_convex, 255)
                        
                        source_face = cv2.bitwise_and(source_image, source_image, mask = source_canvas)
                        
                        rect = cv2.boundingRect(source_convex)
                        
                        subdiv = cv2.Subdiv2D(rect)
                        
                        subdiv.insert(source_landmark_points)
                        
                        triangles = np.array(subdiv.getTriangleList())
                        
                        triangle_indexes = []
                        
                        for triangle_points in triangles:
                            p1 = (triangle_points[0] , triangle_points[1])
                            p2 = (triangle_points[2] , triangle_points[3])
                            p3 = (triangle_points[4] , triangle_points[5])
                            
                            triangle = [np.where((source_landmark_points_arr == p1).all(axis =1))[0][0] , np.where((source_landmark_points_arr == p2).all(axis =1))[0][0], np.where((source_landmark_points_arr == p3).all(axis =1))[0][0]]
                            
                            triangle_indexes.append(triangle)
                        
                    for triangle_index in triangle_indexes:
                        
                        source_triangle = np.array([source_landmark_points[triangle_index[0]], source_landmark_points[triangle_index[1]], source_landmark_points[triangle_index[2]]])
                        
                        source_rect = cv2.boundingRect(source_triangle)
                        
                        x, y , w , h = source_rect
                        
                        source_cropped_rect = source_image[y: y+h , x: x+w]
                        
                        source_triangle_points = np.array([[source_landmark_points[triangle_index[0]][0]-x , source_landmark_points[triangle_index[0]][1]-y] , [source_landmark_points[triangle_index[1]][0]-x , source_landmark_points[triangle_index[1]][1]-y], [source_landmark_points[triangle_index[2]][0]-x , source_landmark_points[triangle_index[2]][1]-y]])
                        
                        des_triangle = np.array([des_landmark_points[triangle_index[0]], des_landmark_points[triangle_index[1]], des_landmark_points[triangle_index[2]]])
                        
                        source_triangle_points = np.float32(source_triangle_points)
                        
                        des_rect = cv2.boundingRect(des_triangle)
                        
                        x, y, w, h = des_rect
                        
                        des_cropped_rect = source_image[y:y+h , x:x+w]
                        
                        des_cropped_rect_mask = np.zeros((h,w), np.uint8)
                        
                        des_triangle_points = np.array([[des_landmark_points[triangle_index[0]][0]-x , des_landmark_points[triangle_index[0]][1]-y] , [des_landmark_points[triangle_index[1]][0]-x , des_landmark_points[triangle_index[1]][1]-y], [des_landmark_points[triangle_index[2]][0]-x , des_landmark_points[triangle_index[2]][1]-y]])
                       
                        
                        cv2.fillConvexPoly(des_cropped_rect_mask, des_triangle_points, 255)
                        
                        des_triangle_points = np.float32(des_triangle_points)
                        
                        mat = cv2.getAffineTransform(source_triangle_points, des_triangle_points)
                        
                        warped_affine_triangle = cv2.warpAffine(source_cropped_rect, mat, (w,h))
                    
                        wraped_affine_triangle = cv2.bitwise_and(warped_affine_triangle, warped_affine_triangle, mask = des_cropped_rect_mask)
                        
                        new_canvas_piece = des_canvas[y:y+h , x:x+w]
                        
                        des_canvas_area = cv2.cvtColor(des_canvas[y:y+h , x:x+w], cv2.COLOR_BGR2GRAY)
                        
                        _ , mask_created_triangle = cv2.threshold(des_canvas_area , 1, 255, cv2.THRESH_BINARY_INV)
                        
                        warped_affine_triangle = cv2.bitwise_and(wraped_affine_triangle, wraped_affine_triangle, mask = mask_created_triangle)
                           
                        new_canvas_piece = cv2.add(new_canvas_piece, warped_affine_triangle)
                        
                        des_canvas[y:y+h, x:x+w] = new_canvas_piece
                    
                    new_des_canvas = np.zeros_like(gray_des)
                    face_mask = cv2.fillConvexPoly(new_des_canvas, des_convex, 255)
                    new_des_canvas = cv2.bitwise_not(face_mask)
                    
                    filled_cover = cv2.bitwise_and(des_image, des_image , mask = new_des_canvas)
                    filled_cover_and_face = cv2.add(filled_cover, des_canvas)
                        
                    x,y,w,h = cv2.boundingRect(des_convex)
                    face_center = (int((x+x+w)/2), int((y+y+h)/2))
                    
                    
                    cloned_face = cv2.seamlessClone(filled_cover_and_face, des_image, face_mask, face_center, cv2.NORMAL_CLONE)
                    
                    cv2.namedWindow("DEEPFAKEOUTPUT", cv2.WINDOW_NORMAL) 
                    
                    resized_output = cv2.resize(cloned_face, (900, 500))  
                    
                    cv2.imshow("DEEPFAKEOUTPUT", resized_output)
                    
                else:
                    cv2.namedWindow("DEEPFAKEOUTPUT", cv2.WINDOW_NORMAL) 
                    
                    resized_output = cv2.resize(des_image, (900, 500))  
                    
                    #cv2.putText(cloned_face,'Hack Projects',(10,500), fontcv2.FONT_HERSHEY_SIMPLEX , 1,(255,255,255),2)
                    
                    cv2.imshow("DEEPFAKEOUTPUT", resized_output)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break   
            
                
            video_output.release() 
            cv2.destroyAllWindows()  
        except:
            print('error: Video error')
 
# df = DeepFake()
# df.source_image("1.jpg") 
# df.destination_image("2.jpg") 
# df.swap_image_faces()