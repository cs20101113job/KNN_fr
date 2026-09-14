# 分別利用KNN/ SVM/ Mediapipe三種模型進行人臉偵測
# 整體流程包含：圖片上傳 --＞格式解碼 --＞ KNN 模型辨識人臉 --＞ 繪製標籤與外框 --＞ Streamlit 視覺化與自動存檔。
import cv2
import numpy as np
import streamlit as st
import os
from PIL import Image
from datetime import datetime
# 載入KNN函式庫
import math
from sklearn import neighbors

import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder


st.title("KNN/ SVM/ Mediapipe三種模型進行人臉偵測")
save_folder = "KNN_SVM_Mediapipe_face_recognition_saved"
os.makedirs(save_folder, exist_ok=True) # os.makedirs() 函數用於創建多層目錄
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
# 將所有圖像處理邏輯，放在 if 裡面
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imgBGR = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # 解碼成 NumPy 陣列 (BGR)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB) # 轉換成 RGB 格式 
    original_img = imgRGB.copy() # 複製給 original_img，位於記憶體中的 RGB 格式 NumPy 陣列
    
    
    #原始上傳圖像
    st.subheader("原始上傳圖像")
    st.image(original_img, use_container_width=True) # st.image() 預設接受 RGB

    # ----KNN人臉偵測(K-Nearest Neighbors，K-近鄰演算法)---

    KNN_bgr = imgBGR.copy()
    KNN_rgb = original_img.copy() # 直接使用已解碼好的陣列
    n_neighbors=None, # n_neighbors投票參考鄰居的數量。
    knn_algo='ball_tree', # knn_algo尋找鄰居的演算法。
    model_path = "trained_knn_model.clf"  # 預訓練模型檔案路徑
    distance_threshold=0.6 # # 距離門檻：大於此數值代表「不認識/陌生人」
    # KNeighborsClassifier(n_neighbors=1) 中（模型分類）
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance') # 使用sklearn庫的neighbors.KNeighborsClassifier類創建KNN分類器對象，n_neighbors指定鄰居數量，algorithm指定尋找鄰居的演算法，weights='distance'表示使用距離加權的方式進行投票，即距離越近的鄰居對分類結果的影響越大
    # 模型已經在記憶體中了，直接拿來做預測
    def predict(KNN_rgb, knn_clf=None, model_path=None, distance_threshold=0.6):
        if knn_clf is None: # 如果模型先前還沒有被載入（也就是 knn_clf 還是 None），程式才會執行接下來的載入動作；如果模型已經存在，就會直接跳過，
            with open(model_path, 'rb') as f: # 以二進位模式讀取已訓練好的 KNN 模型檔案，並將其載入到 knn_clf 變數中。rb 是 read binary 的縮寫，表示以二進位模式讀取檔案。變數 f
                knn_clf = pickle.load(f) # 透過 pickle.load() 讀取已訓練好的 trained_knn_model.clf 檔案，取得預先訓練好的人臉分類器模型。
        

        # 進行人臉座標位置偵測
         # face_locations：目前這張圖片中，所有被偵測到的人臉位置列表。len(face_locations)：偵測到的人臉總數量。
        face_locations = face_recognition.face_locations(KNN_rgb) # 輸出 [(120, 340, 280, 180)] 座標資料，    
        if len(face_locations) == 0: # 圖片中未偵測到任何人臉
            return [] # return [] 代表函式的回傳值是空的列表，表示沒有可供預測的人臉。
        # 提取 128 維人臉特徵向量 (供後續 KNN 辨識模型使用)
    
        faces_encodings = face_recognition.face_encodings(KNN_rgb, known_face_locations=face_locations) # 提取圖像中每個人臉的特徵向量，返回一個包含每個人臉的128維特徵向量的列表，這樣就可以使用這些特徵向量進行人臉識別。
        # 計算特徵距離，使用KNN進行比對
           
            # distance_threshold：人臉識別的歐式距離門檻值（常見設定如 0.6）。距離越小代表兩張臉越相似。
            # are_matches 的結果會是一個布林列表，例如[True, False, True]
            # （搜尋距離）必須先用 kneighbors 算出與最近鄰居的距離
            # knn_clf.kneighbors(..., n_neighbors=1) 中（搜尋距離）
            # n_neighbors=1 表示將 K 值設為 1， 只參考「最靠近的 1 個鄰居」。
        closest_distances, _ = knn_clf.kneighbors(faces_encodings, n_neighbors=1) # knn_clf.kneighbors() 會回傳 (距離矩陣, 索引矩陣) 的元組，計算目前這張臉與訓練資料庫中「最靠近的 1 個鄰居」之間的歐式距離 closest_distances。
        # closest_distances[i][0]，[第 i 張臉][第 0 個鄰居距離]，closest_distances[i] 取得第 $i$ 張臉的資料列（如 [0.35]）。取出的資料加上 [0]，將值從中括號剝出來，拿到純數字 0.35。
        are_matches = [closest_distances[i][0] <= distance_threshold for i in range(len(face_locations))] # are_matches 是一個布林值列表，用於判斷每個人臉特徵向量是否與已知人臉特徵向量匹配，通過比較最近鄰距離是否小於等於距離閾值 distance_threshold 來決定是否匹配，如果匹配則為 True，否則為 False。
        # zip()三合一輸送帶， 函數將三個列表（knn_clf.predict(faces_encodings)KNN 模型硬猜出來的名字、X_face_locations每張人臉在圖片中的座標位置、are_matches有沒有通過距離門檻的布林值）打包成一個可迭代的元組第一組：("小明", (10, 50, 80, 20), True)，這樣就可以同時遍歷這三個列表，對每個人臉進行預測。
        # 幫變數取名字，pred：模型預測的名稱（Prediction），loc：人臉的座標位置（Location），rec：是否辨識成功、認識這個人（Recognized）
        # 真相二選一，if rec else "unknown"：如果 rec 為 True，代表模型認識這個人，則回傳 pred（模型預測的名稱），否則回傳 "unknown"。
        return[(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_locations, are_matches)]    
    # 實際呼叫 predict 函式，並取得 [(人名, (top, right, bottom, left)), ...] 結果
    predictions = predict(KNN_rgb, model_path=model_path, distance_threshold=distance_threshold)

    # 晝框    
    # 讀取 predict() 辨識出的「正確人名 (name)」
    for name, (top, right, bottom, left) in predictions:
        # OpenCV 畫框 (左上點 (left, top)，右下點 (right, bottom))
        cv2.rectangle(KNN_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(KNN_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 將畫好框的 bgr 轉回 RGB，供 Streamlit 用
    KNNresults_rgb = cv2.cvtColor(KNN_bgr, cv2.COLOR_BGR2RGB) 

    # KNN人臉偵測
    st.subheader("KNN人臉偵測")
    st.image(KNNresults_rgb, use_container_width=True) # st.image() 預設接受 RGB

    # KNN自動儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    KNN_filename = os.path.join(save_folder, f"KNN_{timestamp}.png")
    cv2.imwrite(KNN_filename, KNN_bgr)
    st.success(f"KNN人臉偵測已經儲存 {KNN_filename}")
    
   

    
 